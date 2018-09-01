#include "vtkHyperObjectivityFilter.h"
#include <vtkImageData.h>
#include <vtkObjectFactory.h>
#include <vtkStreamingDemandDrivenPipeline.h>
#include <vtkInformationVector.h>
#include <vtkInformation.h>
#include <vtkDataObject.h>
#include <vtkSmartPointer.h>
#include <vtkPointData.h>
#include <vtkDataArray.h>
#include <vtkVector.h>
#include <Eigen\Core>
#include <Eigen\StdVector>
#include <Eigen\QR>
#include <vector>
#include <vtkFloatArray.h>
 
vtkStandardNewMacro(vtkHyperObjectivityFilter);

// Creates a 2x2 matrix from 2 column vectors
static Eigen::Matrix2d make_Matrix2d(const Eigen::Vector2d& c0, const Eigen::Vector2d& c1)
{
	Eigen::Matrix2d M;
	M << c0.x(), c1.x(), c0.y(), c1.y();
	return M;
}

// Create a 2x2 matrix from row-wise components
static Eigen::Matrix2d make_Matrix2d(const double& m00, const double& m01, const double& m10, const double& m11)
{
	Eigen::Matrix2d M;
	M << m00, m01, m10, m11;
	return M;
}


vtkHyperObjectivityFilter::vtkHyperObjectivityFilter() : NeighborhoodU(10), Invariance(AffineInvariance), UseSummedAreaTables(true),
FieldNameV(NULL), FieldNameVx(NULL), FieldNameVy(NULL), FieldNameVt(NULL)
{
	SetFieldNameV("v");
	SetFieldNameVx("vx");
	SetFieldNameVy("vy");
	SetFieldNameVt("vt");
}

int vtkHyperObjectivityFilter::RequestData(vtkInformation *vtkNotUsed(request),
	vtkInformationVector **inputVector,
	vtkInformationVector *outputVector)
{
	using namespace Eigen;
	
	// Get the info objects
	vtkInformation *inInfo = inputVector[0]->GetInformationObject(0);
	vtkInformation *outInfo = outputVector->GetInformationObject(0);

	// Get the input and ouptut
	vtkImageData *input = vtkImageData::SafeDownCast(inInfo->Get(vtkDataObject::DATA_OBJECT()));
	vtkImageData *output = vtkImageData::SafeDownCast(outInfo->Get(vtkDataObject::DATA_OBJECT()));

	// Get the information on the domain
	int* dims = input->GetDimensions();
	double* spacing = input->GetSpacing();
	double* boundsMin = input->GetOrigin();

	// select the system matrix size depending on the chosen invariance
	int systemSize = 6;
	switch (Invariance) {
	default:
	case Objectivity:			systemSize = 6;		break;
	case SimilarityInvariance:	systemSize = 8;		break;
	case AffineInvariance:		systemSize = 12;	break;
	}

	// read the input data and abort if data is not present!
	float* input_v = NULL, *input_vx = NULL, *input_vy = NULL, *input_vt = NULL;
	if (vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameV)) && input->GetPointData()->GetArray(FieldNameV)->GetNumberOfComponents() == 2)
		input_v = vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameV))->GetPointer(0);
	else { cout << "Field " << FieldNameV << " was not found or does not have 2 components!" << endl; return 0; }

	if (vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameVx)) && input->GetPointData()->GetArray(FieldNameVx)->GetNumberOfComponents() == 2)
		input_vx = vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameVx))->GetPointer(0);
	else { cout << "Field " << FieldNameVx << " was not found or does not have 2 components!" << endl; return 0; }

	if (vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameVy)) && input->GetPointData()->GetArray(FieldNameVy)->GetNumberOfComponents() == 2)
		input_vy = vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameVy))->GetPointer(0);
	else { cout << "Field " << FieldNameVy << " was not found or does not have 2 components!" << endl; return 0; }

	if (vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameVt)) && input->GetPointData()->GetArray(FieldNameVt)->GetNumberOfComponents() == 2)
		input_vt = vtkFloatArray::SafeDownCast(input->GetPointData()->GetArray(FieldNameVt))->GetPointer(0);
	else { cout << "Field " << FieldNameVt << " was not found or does not have 2 components!" << endl; return 0; }

	// Create an image to write the result to and copy the input into it (just for initialization)
	vtkSmartPointer<vtkImageData> image = vtkSmartPointer<vtkImageData>::New();
	image->DeepCopy(input);

	vtkDataArray* output_v = image->GetPointData()->GetArray(FieldNameV);
	vtkDataArray* output_vx = image->GetPointData()->GetArray(FieldNameVx);
	vtkDataArray* output_vy = image->GetPointData()->GetArray(FieldNameVy);
	vtkDataArray* output_vt = image->GetPointData()->GetArray(FieldNameVt);

	// Iterate the time steps (in parallel)
#ifdef NDEBUG
#pragma omp parallel for schedule(dynamic,16)
#endif
	for (int it = 0; it < dims[2]; it++)
	{
		// declare matrices for every voxel of a slice
		std::vector<MatrixXd, aligned_allocator<MatrixXd> > _M(dims[0] * dims[1]);
		std::vector<MatrixXd, aligned_allocator<MatrixXd> > _MTM(dims[0] * dims[1]);
		std::vector<MatrixXd, aligned_allocator<MatrixXd> > _MTb(dims[0] * dims[1]);

		// setup the system matrix M and right hand side.
		for (int iy = 0; iy < dims[1]; iy++)
		{
			double y = boundsMin[1] + iy * spacing[1];
			for (int ix = 0; ix < dims[0]; ix++)
			{
				double x = boundsMin[0] + ix * spacing[0];
				int tupleIdx = it * dims[0] * dims[1] + iy*dims[0] + ix;

				// position, velocity and derivatives at the voxel
				Vector2d xx(x, y);
				Vector2d vv(input_v[tupleIdx * 2 + 0], input_v[tupleIdx * 2 + 1]);
				Vector2d dx(input_vx[tupleIdx * 2 + 0], input_vx[tupleIdx * 2 + 1]);
				Vector2d dy(input_vy[tupleIdx * 2 + 0], input_vy[tupleIdx * 2 + 1]);
				Vector2d dt(input_vt[tupleIdx * 2 + 0], input_vt[tupleIdx * 2 + 1]);
				
				// compute 90 degree rotated vectors, setup Jacobian and compute products
				Vector2d Xp(-xx.y(), xx.x());
				Vector2d Vp(-vv.y(), vv.x());
				Matrix2d J = make_Matrix2d(dx, dy);
				Vector2d Jxpvp = -J*Xp + Vp;
				Vector2d Jxv = -J*xx + vv;

				// setup matrix M
				MatrixXd M(2, systemSize);
				switch (Invariance)
				{
				default:
				case Objectivity:
					M(0, 0) = Jxpvp.x(); 	 M(0, 1) = dx.x(); M(0, 2) = dy.x();   M(0, 3) = 1; M(0, 4) = 0;   M(0, 5) = Xp.x();
					M(1, 0) = Jxpvp.y(); 	 M(1, 1) = dx.y(); M(1, 2) = dy.y();   M(1, 3) = 0; M(1, 4) = 1;   M(1, 5) = Xp.y();
					break;
				case SimilarityInvariance:
					M(0, 0) = Jxpvp.x(); 	 M(0, 1) = dx.x(); M(0, 2) = dy.x();   M(0, 3) = 1; M(0, 4) = 0;   M(0, 5) = Xp.x(); M(0, 6) = Jxv.x(); M(0, 7) = xx.x();
					M(1, 0) = Jxpvp.y(); 	 M(1, 1) = dx.y(); M(1, 2) = dy.y();   M(1, 3) = 0; M(1, 4) = 1;   M(1, 5) = Xp.y(); M(1, 6) = Jxv.y();	M(1, 7) = xx.y();
					break;
				case AffineInvariance:
					M(0, 0) = vv.x() - xx.x() * J(0, 0);	M(0, 1) = 0 - xx.x() * J(0, 1);	/**/  M(0, 2) = vv.y() - xx.y() * J(0, 0);  M(0, 3) = 0 - xx.y() * J(0, 1);  /**/  M(0, 4) = J(0, 0);  M(0, 5) = J(0, 1);  /**/  M(0, 6) = 1;  M(0, 7) = 0;  /**/  M(0, 8) = xx.x();  M(0, 9) = 0;       /**/  M(0, 10) = xx.y();  M(0, 11) = 0;
					M(1, 0) = 0 - xx.x() * J(1, 0);	M(1, 1) = vv.x() - xx.x() * J(1, 1);	/**/  M(1, 2) = 0 - xx.y() * J(1, 0);  M(1, 3) = vv.y() - xx.y() * J(1, 1);  /**/  M(1, 4) = J(1, 0);  M(1, 5) = J(1, 1);  /**/  M(1, 6) = 0;	M(1, 7) = 1; /**/  M(1, 8) = 0;       M(1, 9) = xx.x();  /**/  M(1, 10) = 0;       M(1, 11) = xx.y();
					break;
				}

				// store MTM and MTb
				MatrixXd MT = M.transpose();
				_M[iy*dims[0] + ix] = M;
				_MTM[iy*dims[0] + ix] = MT*M;
				_MTb[iy*dims[0] + ix] = MT*dt;
			}
		}

		// compute the prefix sum
		if (UseSummedAreaTables)
		{
			for (int iy = 0; iy < dims[1]; iy++)
				for (int ix = 0; ix < dims[0]; ix++)
				{
					if (ix > 0) {
						_MTM[iy*dims[0] + ix] += _MTM[iy*dims[0] + (ix - 1)];
						_MTb[iy*dims[0] + ix] += _MTb[iy*dims[0] + (ix - 1)];
					}
					if (iy > 0) {
						_MTM[iy*dims[0] + ix] += _MTM[(iy - 1)*dims[0] + ix];
						_MTb[iy*dims[0] + ix] += _MTb[(iy - 1)*dims[0] + ix];
					}
					if (ix > 0 && iy > 0) {
						_MTM[iy*dims[0] + ix] -= _MTM[(iy - 1)*dims[0] + (ix - 1)];
						_MTb[iy*dims[0] + ix] -= _MTb[(iy - 1)*dims[0] + (ix - 1)];
					}
				}
		}
		
		// solve the system for each pixel
		for (int iy = 0; iy < dims[1]; iy++)
		{
			double y = boundsMin[1] + iy * spacing[1];
			for (int ix = 0; ix < dims[0]; ix++)
			{
				double x = boundsMin[0] + ix * spacing[0];
				int tupleIdx = it * dims[0] * dims[1] + iy*dims[0] + ix;

				// corner indices of the neighborhood region
				int x1 = std::min(std::max(0, ix - NeighborhoodU - 1), dims[0] - 1);
				int y1 = std::min(std::max(0, iy - NeighborhoodU - 1), dims[1] - 1);
				int x2 = std::min(std::max(0, ix + NeighborhoodU), dims[0] - 1);
				int y2 = std::min(std::max(0, iy + NeighborhoodU), dims[1] - 1);

				// compute the sum of all matrices/vectors in neighborhood region
				MatrixXd MTM(systemSize, systemSize);
				MTM.setZero();
				VectorXd MTb(systemSize, 1);
				MTb.setZero();

				if (UseSummedAreaTables)
				{
					MTM = _MTM[y2*dims[0] + x2] + _MTM[y1*dims[0] + x1] - _MTM[y1*dims[0] + x2] - _MTM[y2*dims[0] + x1];
					MTb = _MTb[y2*dims[0] + x2] + _MTb[y1*dims[0] + x1] - _MTb[y1*dims[0] + x2] - _MTb[y2*dims[0] + x1];
				}
				else
				{
					for (int wy = y1; wy <= y2; ++wy)
						for (int wx = x1; wx <= x2; ++wx)
						{
							MTM += _MTM[wy*dims[0] + wx];
							MTb += _MTb[wy*dims[0] + wx];
						}
				}
				
				// solve for reference frame parameters
				VectorXd uu = MTM.fullPivHouseholderQr().solve(MTb);
				
				// compute new vector field in optimal frame
				Vector2d xx(x, y);
				Vector2d vv(input_v[tupleIdx * 2 + 0], input_v[tupleIdx * 2 + 1]);
				Vector2d dx(input_vx[tupleIdx * 2 + 0], input_vx[tupleIdx * 2 + 1]);
				Vector2d dy(input_vy[tupleIdx * 2 + 0], input_vy[tupleIdx * 2 + 1]);
				Vector2d dt(input_vt[tupleIdx * 2 + 0], input_vt[tupleIdx * 2 + 1]);
				Matrix2d J = make_Matrix2d(dx, dy);
				Vector2d vnew(0, 0);
				Matrix2d Jnew = make_Matrix2d(0, 0, 0, 0);
				Vector2d Xp(-xx.y(), xx.x());
				switch (Invariance)
				{
				case Objectivity:
					vnew = vv + Vector2d(uu(1), uu(2)) - uu(0) * Xp;
					Jnew = J + make_Matrix2d(0, uu(0), -uu(0), 0);
					break;
				case SimilarityInvariance:
					vnew = vv + Vector2d(uu(1), uu(2)) - uu(0) * Xp - uu(6) * xx;
					Jnew = J + make_Matrix2d(0, uu(0), -uu(0), 0) - make_Matrix2d(uu(6), 0, 0, uu(6));
					break;
				case AffineInvariance:
					Matrix2d H1 = make_Matrix2d(-uu(0), -uu(2), -uu(1), -uu(3));
					Vector2d k1(uu(4), uu(5));
					vnew = vv + H1 * xx + k1;
					Jnew = J + H1;
					break;
				}
				Vector2d vtnew = dt - _M[iy * dims[0] + ix] * uu;

				// store the result
				output_v->SetTuple2(tupleIdx, vnew.x(), vnew.y());
				output_vx->SetTuple2(tupleIdx, Jnew(0,0), Jnew(1,0));
				output_vy->SetTuple2(tupleIdx, Jnew(0,1), Jnew(1,1));
				output_vt->SetTuple2(tupleIdx, vtnew.x(), vtnew.y());
			}
		}
	}

	// Copy the computed image to the output
	output->ShallowCopy(image);

	// Update the extent
	int extent[6];
	input->GetExtent(extent);
	output->SetExtent(extent);
	outInfo->Set(vtkStreamingDemandDrivenPipeline::UPDATE_EXTENT(), extent, 6);
	outInfo->Set(vtkStreamingDemandDrivenPipeline::WHOLE_EXTENT(), extent, 6);
	return 1;
}