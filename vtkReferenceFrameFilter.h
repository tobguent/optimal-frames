#ifndef __vtkImageAlgorithmFilter_h
#define __vtkImageAlgorithmFilter_h
 
#include "vtkImageAlgorithm.h"
 
class vtkReferenceFrameFilter : public vtkImageAlgorithm
{
public:
	enum EInvariance
	{
		Objectivity,
		SimilarityInvariance,
		AffineInvariance,
		Displacement
	};
	static vtkReferenceFrameFilter *New();
	vtkTypeMacro(vtkReferenceFrameFilter, vtkImageAlgorithm);

	vtkReferenceFrameFilter();

	vtkGetMacro(NeighborhoodU, int);
	vtkSetMacro(NeighborhoodU, int);

	vtkGetMacro(Invariance, EInvariance);
	vtkSetMacro(Invariance, EInvariance);

	vtkGetMacro(UseSummedAreaTables, bool);
	vtkSetMacro(UseSummedAreaTables, bool);

	vtkGetMacro(TaylorOrder, int);
	vtkSetMacro(TaylorOrder, int);

	vtkGetStringMacro(FieldNameV);
	vtkSetStringMacro(FieldNameV);
	
	vtkGetStringMacro(FieldNameVx);
	vtkSetStringMacro(FieldNameVx);

	vtkGetStringMacro(FieldNameVy);
	vtkSetStringMacro(FieldNameVy);

	vtkGetStringMacro(FieldNameVt);
	vtkSetStringMacro(FieldNameVt);

protected:

	int RequestData(vtkInformation *, vtkInformationVector **, vtkInformationVector *);
	int NeighborhoodU;
	EInvariance Invariance;
	bool UseSummedAreaTables;
	int TaylorOrder;
	char* FieldNameV;
	char* FieldNameVx;
	char* FieldNameVy;
	char* FieldNameVt;

private:
	vtkReferenceFrameFilter(const vtkReferenceFrameFilter&);  // Not implemented.
	void operator=(const vtkReferenceFrameFilter&);  // Not implemented.

};
 
#endif