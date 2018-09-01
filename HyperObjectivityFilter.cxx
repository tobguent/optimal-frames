#include <vtkSmartPointer.h>
#include <vtkMath.h>
#include <vtkImageData.h>
#include <vtkPointData.h>
#include <vtkVector.h>
#include <vtkXMLImageDataReader.h>
#include <vtkXMLImageDataWriter.h>
#include <vtkImageReader.h>
#include <vtkFloatArray.h>
#include "vtkHyperObjectivityFilter.h"

// Reads a vector field from file or uses a Stuart vortex by default.
vtkSmartPointer<vtkImageData> InitializeTestCase(int argc, char *argv[], int* neighborhoodU, vtkHyperObjectivityFilter::EInvariance* invariance);

// Input arguments:
//   [1] : path to the vti-file in XML format, containing the unsteady vector field and all its first-order derivatives as floats in 2D space-time (the third dimension is time, the vectors stored the xy-components only)
//   [2] : neighborhood size (int)
//   [3] : invariance choice (0=objectivity, 1=similarity invariance, 2=affine invariance)
int main(int argc, char *argv[])
{
	// ---------------------------------------------
	// -------- Initialization of test case --------
	// ---------------------------------------------
	cout << "Initialize test case..." << endl;

	int neighborhoodU;
	vtkHyperObjectivityFilter::EInvariance invariance;
	vtkSmartPointer<vtkImageData> input = InitializeTestCase(argc, argv, &neighborhoodU, &invariance);

	// ---------------------------------------------
	// --- Compute vector field in optimal frame ---
	// ---------------------------------------------
	cout << "Computing optimal reference frame..." << endl;

	vtkSmartPointer<vtkHyperObjectivityFilter> filter = vtkSmartPointer<vtkHyperObjectivityFilter>::New();
	filter->SetInputData(input);
	filter->SetNeighborhoodU(neighborhoodU);	// size of neighborhood region U in voxels
	filter->SetInvariance(invariance);			// select the desired invariance
	filter->SetFieldNameV("v");					// name of the point data field that contains velocities
	filter->SetFieldNameVx("vx");				// name of the point data field that contains x-partials of the velocity
	filter->SetFieldNameVy("vy");				// name of the point data field that contains y-partials of the velocity
	filter->SetFieldNameVt("vt");				// name of the point data field that contains t-partials of the velocity
	filter->Update();
	vtkImageData* output = filter->GetOutput();

	// ---------------------------------------------
	// ----------- Write result to file ------------
	// ---------------------------------------------
	vtkSmartPointer<vtkXMLImageDataWriter> writer = vtkSmartPointer<vtkXMLImageDataWriter>::New();
	writer->SetFileName("v_optimal.vti");
	writer->SetInputData(output);
	writer->Update();

	cout << "Done." << endl;
	return 0;
}

// This function samples a Stuart vortex vector field and returns the flow on a discrete grid.
vtkSmartPointer<vtkImageData> CreateStuartVectorField()
{
	// Set resolution and physical extent of the domain
	int dims[] = { 64, 64, 64 };
	double boundsMin[] = { -4.0, -2.0, 0.0 };
	double boundsMax[] = { 4.0, 2.0, 2.0*vtkMath::Pi() };
	double spacing[] = {
		(boundsMax[0] - boundsMin[0]) / ((double)dims[0] - 1.0),
		(boundsMax[1] - boundsMin[1]) / ((double)dims[1] - 1.0),
		(boundsMax[2] - boundsMin[2]) / ((double)dims[2] - 1.0)
	};

	// create arrays to the store the flow and its derivatives
	vtkSmartPointer<vtkFloatArray> array_v = vtkSmartPointer<vtkFloatArray>::New();
	vtkSmartPointer<vtkFloatArray> array_vx = vtkSmartPointer<vtkFloatArray>::New();
	vtkSmartPointer<vtkFloatArray> array_vy = vtkSmartPointer<vtkFloatArray>::New();
	vtkSmartPointer<vtkFloatArray> array_vt = vtkSmartPointer<vtkFloatArray>::New();
	array_v->SetNumberOfComponents(2);
	array_vx->SetNumberOfComponents(2);
	array_vy->SetNumberOfComponents(2);
	array_vt->SetNumberOfComponents(2);
	array_v->SetNumberOfTuples(dims[0] * dims[1] * dims[2]);
	array_vx->SetNumberOfTuples(dims[0] * dims[1] * dims[2]);
	array_vy->SetNumberOfTuples(dims[0] * dims[1] * dims[2]);
	array_vt->SetNumberOfTuples(dims[0] * dims[1] * dims[2]);
	array_v->SetName("v");
	array_vx->SetName("vx");
	array_vy->SetName("vy");
	array_vt->SetName("vt");

	// Sample the test vector field in 2D space-time
	for (int it = 0; it < dims[2]; it++)
	{
		double t = boundsMin[2] + it * spacing[2];
		for (int iy = 0; iy < dims[1]; iy++)
		{
			double y = boundsMin[1] + iy * spacing[1];
			for (int ix = 0; ix < dims[0]; ix++)
			{
				double x = boundsMin[0] + ix * spacing[0];
				int tupleIdx = it*dims[0] * dims[1] + iy*dims[0] + ix;
				array_v->SetTuple2(tupleIdx, sinh(y) / (cosh(y) - 0.25*cos(x - t)) + 1, -(0.25*sin(x - t)) / (cosh(y) - 0.25*cos(x - t)));
				array_vx->SetTuple2(tupleIdx, -(0.25*sin(x - t)*sinh(y)) / ((cosh(y) - 0.25*cos(x - t))*(cosh(y) - 0.25*cos(x - t))), (0.0625*(sin(x - t)*sin(x - t))) / ((cosh(y) - 0.25*cos(x - t))*(cosh(y) - 0.25*cos(x - t))) - (0.25*cos(x - t)) / (cosh(y) - 0.25*cos(x - t)));
				array_vy->SetTuple2(tupleIdx, cosh(y) / (cosh(y) - 0.25*cos(x - t)) - (sinh(y)*sinh(y)) / ((cosh(y) - 0.25*cos(x - t))*(cosh(y) - 0.25*cos(x - t))), (0.25*sin(x - t)*sinh(y)) / ((cosh(y) - 0.25*cos(x - t))*(cosh(y) - 0.25*cos(x - t))));
				array_vt->SetTuple2(tupleIdx, (0.25*sin(x - t)*sinh(y)) / ((cosh(y) - 0.25*cos(x - t))*(cosh(y) - 0.25*cos(x - t))), (0.25*cos(x - t)) / (cosh(y) - 0.25*cos(x - t)) - (0.0625*(sin(x - t)*sin(x - t))) / ((cosh(y) - 0.25*cos(x - t))*(cosh(y) - 0.25*cos(x - t))));
			}
		}
	}

	// Allocate memory for the vector field in 2D space-time (2D + time = 3D)
	vtkSmartPointer<vtkImageData> input = vtkSmartPointer<vtkImageData>::New();
	input->SetDimensions(dims);
	input->SetSpacing(spacing);
	input->SetOrigin(boundsMin);
	input->GetPointData()->AddArray(array_v);
	input->GetPointData()->AddArray(array_vx);
	input->GetPointData()->AddArray(array_vy);
	input->GetPointData()->AddArray(array_vt);
	return input;
}

// Reads a vector field from file or uses a Stuart vortex by default.
vtkSmartPointer<vtkImageData> InitializeTestCase(int argc, char *argv[], int* neighborhoodU, vtkHyperObjectivityFilter::EInvariance* invariance)
{
	// set the default parameters
	*neighborhoodU = 10;										// size of neighborhood region in voxels: [2*N+1]^2
	*invariance = vtkHyperObjectivityFilter::AffineInvariance;	// selected invariance

	// read command line arguments to create the input vector field
	vtkSmartPointer<vtkImageData> input;
	if (argc == 4) 
	{
		// create a reader and try to read the file
		vtkSmartPointer<vtkXMLImageDataReader> reader = vtkSmartPointer<vtkXMLImageDataReader>::New();
		reader->SetFileName(argv[1]);
		reader->Update();
		input = reader->GetOutput();
		
		// if the reader read a plausible dimension, we assume that the reader was successful.
		if (input->GetDimensions()[0] != 0) {
			cout << "  Data set \t = " << argv[1] << endl;
			// read additional parameters: neighborhood size and invariance
			*neighborhoodU = atoi(argv[2]);
			switch (atoi(argv[3])) {
			case 0: *invariance = vtkHyperObjectivityFilter::Objectivity; break;
			case 1: *invariance = vtkHyperObjectivityFilter::SimilarityInvariance; break;
			case 2: *invariance = vtkHyperObjectivityFilter::AffineInvariance; break;
			}
		}
		else // if unsuccessful, for instance when file was not found, build a stuart vortex instead.
		{
			cout << "  Error reading vti file (XML reader): " << argv[1] << endl;
			cout << "  Using Stuart vortex instead with default parameters." << endl;
			input = CreateStuartVectorField();
			cout << "  Data set \t = Stuart Vortex" << endl;
		}	
	}
	else 
	{
		// if no command line arguments were specified, use a simple Stuart vortex vector field
		input = CreateStuartVectorField();	
		cout << "  Data set \t = Stuart Vortex" << endl;
	}
	cout << "  Neighborhood U = " << *neighborhoodU << endl;
	cout << "  Invariance \t = " << (*invariance == 0 ? "Objectivity" : (*invariance == 1 ? "Similarity Invariance" : "Affine Invariance")) << endl;
	return input;
}