#ifndef __vtkObjectivityFilter_h
#define __vtkObjectivityFilter_h
 
#include "vtkImageAlgorithm.h"
 
class vtkObjectivityFilter : public vtkImageAlgorithm
{
public:
	
	static vtkObjectivityFilter *New();
	vtkTypeMacro(vtkObjectivityFilter, vtkImageAlgorithm);

	vtkObjectivityFilter();

	vtkGetMacro(NeighborhoodU, int);
	vtkSetMacro(NeighborhoodU, int);

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
	bool UseSummedAreaTables;
	char* FieldNameV;
	char* FieldNameVx;
	char* FieldNameVy;
	char* FieldNameVt;

private:
	vtkObjectivityFilter(const vtkObjectivityFilter&);  // Not implemented.
	void operator=(const vtkObjectivityFilter&);  // Not implemented.

};
 
#endif