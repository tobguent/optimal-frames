#ifndef __vtkImageAlgorithmFilter_h
#define __vtkImageAlgorithmFilter_h
 
#include "vtkImageAlgorithm.h"
 
class vtkHyperObjectivityFilter : public vtkImageAlgorithm
{
public:
	enum EInvariance
	{
		Objectivity,
		SimilarityInvariance,
		AffineInvariance
	};
	static vtkHyperObjectivityFilter *New();
	vtkTypeMacro(vtkHyperObjectivityFilter, vtkImageAlgorithm);

	vtkHyperObjectivityFilter();

	vtkGetMacro(NeighborhoodU, int);
	vtkSetMacro(NeighborhoodU, int);

	vtkGetMacro(Invariance, EInvariance);
	vtkSetMacro(Invariance, EInvariance);

	vtkGetMacro(UseSummedAreaTables, bool);
	vtkSetMacro(UseSummedAreaTables, bool);

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
	char* FieldNameV;
	char* FieldNameVx;
	char* FieldNameVy;
	char* FieldNameVt;

private:
	vtkHyperObjectivityFilter(const vtkHyperObjectivityFilter&);  // Not implemented.
	void operator=(const vtkHyperObjectivityFilter&);  // Not implemented.

};
 
#endif