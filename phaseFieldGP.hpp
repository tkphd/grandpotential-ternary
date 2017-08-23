std::string PROGRAM = "GP";
std::string MESSAGE = "Grand Potential Cr-Nb-Ni phase transformation code";

typedef MMSP::grid<1,MMSP::vector<double> > GRID1D;
typedef MMSP::grid<2,MMSP::vector<double> > GRID2D;
typedef MMSP::grid<3,MMSP::vector<double> > GRID3D;

template<typename T>
void computeCompositions(MMSP::vector<T>& v);

template<int dim, typename T>
T computeEnergy(MMSP::grid<dim,MMSP::vector<T> >& GRID);

template <typename T> T g(const T& x) {return x * x * x * (6. * x * x - 15. * x + 10.);}
template <typename T> T gprime(const T& x) {return 30. * x * x * (x - 1.) * (x - 1.);}

template <typename T> T fdw(const T& x) {return x*x*(1.-x)*(1.-x);}
template <typename T> T fprime(const T& x) {return 2.*x*(x-1.)*(2.*x - 1.);}

template <int dim, typename T> MMSP::vector<T> ranged_laplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, int start, int end);
template <int dim, typename T> MMSP::vector<T> divGradCr(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, int N);
template <int dim, typename T> MMSP::vector<T> divGradNb(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, int N);
