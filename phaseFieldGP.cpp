/*************************************************************************************
 * File: phaseFieldGP.cpp                                                            *
 * Algorithms for 2D and 3D isotropic Cr-Nb-Ni alloy phase transformations           *
 * using grand potential formalism after Plapp. Derivation thanks to N. Ofori-Opoku  *
 *                                                                                   *
 * Questions/comments to trevor.keller@nist.gov (Trevor Keller, Ph.D.)               *
 *                                                                                   *
 * This software was developed at the National Institute of Standards and Technology *
 * by employees of the Federal Government in the course of their official duties.    *
 * Pursuant to title 17 section 105 of the United States Code this software is not   *
 * subject to copyright protection and is in the public domain. NIST assumes no      *
 * responsibility whatsoever for the use of this code by other parties, and makes no *
 * guarantees, expressed or implied, about its quality, reliability, or any other    *
 * characteristic. We would appreciate acknowledgement if the software is used.      *
 *                                                                                   *
 * This software can be redistributed and/or modified freely provided that any       *
 * derivative works bear some notice that they are derived from it, and any modified *
 * versions bear some notice that they have been modified.                           *
 *************************************************************************************/

#include <cmath>
#include <random>
#include <sstream>
#include <vector>
#include <omp.h>
#include "MMSP.hpp"
#include "phaseFieldGP.hpp"

// numerical parameters
const double meshres = 5.0e-9;
const double dt = 7.5e-5;

// phase-field parameters
const double sigma = 1.01;
const double kappa = 1.24e-8;
const double omega = (3.0 * 2.2 * sigma) / (10. * meshres);
const double Lmob = 2.904e-11;
const double alpha = 1.07e11;

// equilibrium concentrations
const double xe_Cr_gam = 0.490;
const double xe_Nb_gam = 0.025;
const double xe_Cr_del = 0.015;
const double xe_Nb_del = 0.245;
const double xe_Cr_lav = 0.300;
const double xe_Nb_lav = 0.328;

// system composition
const double x0_Cr = 0.45;
const double x0_Nb = 0.07;

// diffusion constants
const double D_CrCr = 2.16e-15;
const double D_CrNb = 0.56e-15;
const double D_NbCr = 2.96e-15;
const double D_NbNb = 4.29e-15;

const double D_Cr[2] = {D_CrCr, D_CrNb}; // first column of diffusivity matrix
const double D_Nb[2] = {D_NbCr, D_NbNb}; // second column

// paraboloid curvatures (from CALPHAD)
const double gam_A_CrCr =   4948774782.83651;
const double gam_A_NbNb =  75540884134.00040;
const double gam_A_CrNb =  15148919424.32730;

const double del_A_CrCr =  64657963775.51020;
const double del_A_CrNb =  16970167763.73330;
const double del_A_NbNb = 203631595326.53000;

const double lav_A_CrCr =  20981468851.01740;
const double lav_A_CrNb =  -3224431198.33485;
const double lav_A_NbNb = 305107785439.43300;

// grand potential susceptibilities (Eqn. 23)
const double gam_X_CrCr =  gam_A_NbNb / (gam_A_CrCr * gam_A_NbNb - gam_A_CrNb * gam_A_CrNb);
const double del_X_CrCr =  del_A_NbNb / (del_A_CrCr * del_A_NbNb - del_A_CrNb * del_A_CrNb);
const double lav_X_CrCr =  lav_A_NbNb / (lav_A_CrCr * lav_A_NbNb - lav_A_CrNb * lav_A_CrNb);

const double gam_X_CrNb = -gam_A_CrNb / (gam_A_CrCr * gam_A_NbNb - gam_A_CrNb * gam_A_CrNb);
const double del_X_CrNb = -del_A_CrNb / (del_A_CrCr * del_A_NbNb - del_A_CrNb * del_A_CrNb);
const double lav_X_CrNb = -lav_A_CrNb / (lav_A_CrCr * lav_A_NbNb - lav_A_CrNb * lav_A_CrNb);

const double gam_X_NbCr = -gam_A_CrNb / (gam_A_CrCr * gam_A_NbNb - gam_A_CrNb * gam_A_CrNb);
const double del_X_NbCr = -del_A_CrNb / (del_A_CrCr * del_A_NbNb - del_A_CrNb * del_A_CrNb);
const double lav_X_NbCr = -lav_A_CrNb / (lav_A_CrCr * lav_A_NbNb - lav_A_CrNb * lav_A_CrNb);

const double gam_X_NbNb =  gam_A_CrCr / (gam_A_CrCr * gam_A_NbNb - gam_A_CrNb * gam_A_CrNb);
const double del_X_NbNb =  del_A_CrCr / (del_A_CrCr * del_A_NbNb - del_A_CrNb * del_A_CrNb);
const double lav_X_NbNb =  lav_A_CrCr / (lav_A_CrCr * lav_A_NbNb - lav_A_CrNb * lav_A_CrNb);

const double gam_X_Cr[2] = {gam_X_CrCr, gam_X_CrNb};
const double gam_X_Nb[2] = {gam_X_NbCr, gam_X_NbNb};

const double del_X_Cr[2] = {del_X_CrCr, del_X_CrNb};
const double del_X_Nb[2] = {del_X_NbCr, del_X_NbNb};

const double lav_X_Cr[2] = {lav_X_CrCr, lav_X_CrNb};
const double lav_X_Nb[2] = {lav_X_NbCr, lav_X_NbNb};

namespace MMSP {

void generate(int dim, const char* filename)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	if (dim==2) {
		// set mesh size and resolution
		const int Nx = 320;
		const int Ny = 192;
		double dV = 1.0;
		double Ntot = 1.0;

		// construct grid
		const int Nfields = 12;
		GRID2D initGrid(Nfields, -Nx/2, Nx/2, -Ny/2, Ny/2);

		// boundary conditions
		for (int d = 0; d < dim; d++) {
			dx(initGrid,d)=meshres;
			dV *= meshres;
			Ntot *= g1(initGrid, d) - g0(initGrid, d);
			if (x0(initGrid, d) == g0(initGrid, d))
				b0(initGrid, d) = Neumann;
			if (x1(initGrid, d) == g1(initGrid, d))
				b1(initGrid, d) = Neumann;
		}

		// initial conditions
		const int R = 20; // seed radius, in mesh points
		const int D = 18 + Nx/2; // seed separation, in mesh points
		const double Np = 2.0 * M_PI * R * R; // total area of precipitates
		const double Nt = Nx * Ny; // total system size
		const double xCr0 = (x0_Cr * Nt - (xe_Cr_del + xe_Cr_lav) * Np) / (Nt - Np);
		const double xNb0 = (x0_Nb * Nt - (xe_Nb_del + xe_Nb_lav) * Np) / (Nt - Np);

		// initial deviation from equilibrium compositions:
		const double dc_Cr_gam = xCr0 - xe_Cr_gam;
		const double dc_Nb_gam = xNb0 - xe_Nb_gam;

		const double dc_Cr_del = 0.;
		const double dc_Nb_del = 0.;

		const double dc_Cr_lav = 0.;
		const double dc_Nb_lav = 0.;

		// initial grand potential deviations
		const double u_Cr_gam = gam_A_CrCr * dc_Cr_gam + gam_A_CrNb * dc_Nb_gam;
		const double u_Nb_gam = gam_A_CrNb * dc_Cr_gam + gam_A_NbNb * dc_Nb_gam;

		const double u_Cr_del = del_A_CrCr * dc_Cr_del + del_A_CrNb * dc_Nb_del;
		const double u_Nb_del = del_A_CrNb * dc_Cr_del + del_A_NbNb * dc_Nb_del;

		const double u_Cr_lav = lav_A_CrCr * dc_Cr_lav + lav_A_CrNb * dc_Nb_lav;
		const double u_Nb_lav = lav_A_CrNb * dc_Cr_lav + lav_A_NbNb * dc_Nb_lav;

		const vector<double> blank(Nfields, 0.);

		double energy = 0.;
		FILE* efile;
		if (rank==0)
			efile = fopen("energy.csv", "w");

		#pragma omp parallel for
		for (int n = 0; n < nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			initGrid(n) = blank;

			int r = sqrt((abs(x[0]) - D/2) * (abs(x[0]) - D/2) + x[1] * x[1]);

			if (r < R) { // point lies within a precipitate
				if (x[0] < 0) { // delta precipitate
					initGrid(n)[2] = 1.0;
				} else { // Laves precipitate
					initGrid(n)[3] = 1.0;
				}
			}

			const double phi_del = initGrid(n)[2];
			const double phi_lav = initGrid(n)[3];
			const double phi_gam = 1. - phi_del - phi_lav;

			// set chemical potentials
			initGrid(n)[0] = u_Cr_del * phi_del + u_Cr_lav * phi_lav + u_Cr_gam * phi_gam;
			initGrid(n)[1] = u_Nb_del * phi_del + u_Nb_lav * phi_lav + u_Nb_gam * phi_gam;

			// set compositions
			computeCompositions(initGrid(n));
		}

		ghostswap(initGrid);

		#pragma omp parallel for
		for (int n = 0; n < nodes(initGrid); n++) {
			vector<int> x = position(initGrid, n);
			const vector<double>& initGridN = initGrid(n);
			const double& mu_Cr = initGridN[0];
			const double& mu_Nb = initGridN[1];
			const double& phi_del = initGridN[2];
			const double& phi_lav = initGridN[3];
			const double phi_gam = 1. - phi_del - phi_lav;

			vector<double> grad_phi_del = gradient(initGrid, x, 2);
			vector<double> grad_phi_lav = gradient(initGrid, x, 3);

			// compute grand potentials (Eqn. 19)
			const double w_gam = - (mu_Cr * mu_Cr * gam_X_CrCr)/2. - (mu_Nb * mu_Nb * gam_X_NbNb) / 2. - (mu_Cr * mu_Nb * gam_X_CrNb) - mu_Cr * xe_Cr_gam - mu_Nb * xe_Nb_gam;
			const double w_del = - (mu_Cr * mu_Cr * del_X_CrCr)/2. - (mu_Nb * mu_Nb * del_X_NbNb) / 2. - (mu_Cr * mu_Nb * del_X_CrNb) - mu_Cr * xe_Cr_del - mu_Nb * xe_Nb_del;
			const double w_lav = - (mu_Cr * mu_Cr * lav_X_CrCr)/2. - (mu_Nb * mu_Nb * lav_X_NbNb) / 2. - (mu_Cr * mu_Nb * lav_X_CrNb) - mu_Cr * xe_Cr_lav - mu_Nb * xe_Nb_lav;

			double myenergy = 0.;
			myenergy += 0.5 * kappa * grad_phi_del * grad_phi_del + omega * fdw(phi_del) + alpha * phi_del*phi_del * phi_lav*phi_lav;
			myenergy += 0.5 * kappa * grad_phi_lav * grad_phi_lav + omega * fdw(phi_lav) + alpha * phi_del*phi_del * phi_lav*phi_lav;
			myenergy += g(phi_gam) * w_gam + g(phi_del) * w_del + g(phi_lav) * w_lav;

			myenergy *= dV;

			#pragma omp atomic
			energy += myenergy;
		}

		if (rank==0) {
			fprintf(efile, "time,energy\n");
			fprintf(efile, "0,%f\n", energy);
			fclose(efile);
		}

		// write intial checkpoint to disk
		output(initGrid,filename);

	} else {
		if (rank==0)
			std::cerr << "ERROR: " << dim << "-D grids are not implemented." << std::endl;
		MMSP::Abort(-1);
	}
}

template <int dim, typename T> void update(grid<dim,vector<T> >& oldGrid, int steps)
{
	int rank = 0;
	#ifdef MPI_VERSION
	rank = MPI::COMM_WORLD.Get_rank();
	#endif

	double dV = 1.;
	for (int d=0; d<dim; d++)
		dV *= dx(oldGrid, d);

	ghostswap(oldGrid);

	grid<dim,vector<T> > newGrid(oldGrid);

	static double elapsed = 0.;

	FILE* efile;
	if (rank==0)
		efile = fopen("energy.csv", "a");

	for (int step = 0; step < steps; step++) {
		if (rank == 0)
			print_progress(step, steps);

		double energy = 0.;

		#pragma omp parallel for
		for (int n = 0; n < nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n);
			const vector<T>& oldGridN = oldGrid(n);

			const T& mu_Cr = oldGridN[0];
			const T& mu_Nb = oldGridN[1];

			const T& phi_del = oldGridN[2];
			const T& phi_lav = oldGridN[3];

			const T& gam_xcr = oldGridN[4];
			const T& gam_xnb = oldGridN[5];
			const T& del_xcr = oldGridN[6];
			const T& del_xnb = oldGridN[7];
			const T& lav_xcr = oldGridN[8];
			const T& lav_xnb = oldGridN[9];

			const T g_del = g(phi_del);
			const T g_lav = g(phi_lav);
			const T g_gam = 1. - g_del - g_lav;

			// compute grand potentials (Eqn. 19)
			const T w_gam = - (mu_Cr * mu_Cr * gam_X_CrCr)/2. - (mu_Nb * mu_Nb * gam_X_NbNb) / 2. - (mu_Cr * mu_Nb * gam_X_CrNb) - mu_Cr * xe_Cr_gam - mu_Nb * xe_Nb_gam;
			const T w_del = - (mu_Cr * mu_Cr * del_X_CrCr)/2. - (mu_Nb * mu_Nb * del_X_NbNb) / 2. - (mu_Cr * mu_Nb * del_X_CrNb) - mu_Cr * xe_Cr_del - mu_Nb * xe_Nb_del;
			const T w_lav = - (mu_Cr * mu_Cr * lav_X_CrCr)/2. - (mu_Nb * mu_Nb * lav_X_NbNb) / 2. - (mu_Cr * mu_Nb * lav_X_CrNb) - mu_Cr * xe_Cr_lav - mu_Nb * xe_Nb_lav;

			// compute sources of chemical potential
			const vector<T> lap_phi = ranged_laplacian(oldGrid, x, 2, 4);
			const vector<T> lap_Cr = divGradCr(oldGrid, x, 2);
			const vector<T> lap_Nb = divGradNb(oldGrid, x, 2);

			// compute inverse susceptibilities
			const T inv_X_CrCr = g_gam * gam_A_CrCr + g_del * del_A_CrCr + g_lav * lav_A_CrCr;
			const T inv_X_CrNb = g_gam * gam_A_CrNb + g_del * del_A_CrNb + g_lav * lav_A_CrNb;
			const T inv_X_NbNb = g_gam * gam_A_NbNb + g_del * del_A_NbNb + g_lav * lav_A_NbNb;

			// phi equation of motion
			const T dphidt_del = Lmob * ( kappa * lap_phi[0]
			                            - omega * fprime(phi_del)
			                            - 2. * alpha * phi_del * phi_lav * phi_lav
			                            - (w_del - w_gam) * gprime(phi_del)
			                   );
			const T dphidt_lav = Lmob * ( kappa * lap_phi[1]
			                            - omega * fprime(phi_lav)
			                            - 2. * alpha * phi_lav * phi_del * phi_del
			                            - (w_lav - w_gam) * gprime(phi_lav)
			                   );

			newGrid(n)[2] = phi_del + dt * dphidt_del;
			newGrid(n)[3] = phi_lav + dt * dphidt_lav;

			// mu equation of motion
			const T dudt_A = lap_Cr[0] + lap_Cr[1] - (gprime(phi_del) * (del_xcr - gam_xcr) * dphidt_del - gprime(phi_lav) * (lav_xcr - gam_xcr) * dphidt_lav);
			const T dudt_B = lap_Nb[0] + lap_Nb[1] - (gprime(phi_del) * (del_xnb - gam_xnb) * dphidt_del - gprime(phi_lav) * (lav_xnb - gam_xnb) * dphidt_lav);
			const T dudt_Cr = inv_X_CrCr * dudt_A + inv_X_CrNb * dudt_B;
			const T dudt_Nb = inv_X_CrNb * dudt_A + inv_X_NbNb * dudt_B;

			newGrid(n)[0] = mu_Cr + dt * dudt_Cr;
			newGrid(n)[1] = mu_Nb + dt * dudt_Nb;

			computeCompositions(newGrid(n));
		}

		elapsed += dt;
		swap(oldGrid, newGrid);
		ghostswap(oldGrid);

		#pragma omp parallel for
		for (int n = 0; n < nodes(oldGrid); n++) {
			vector<int> x = position(oldGrid, n);
			const vector<T>& oldGridN = oldGrid(n);
			const T& mu_Cr = oldGridN[0];
			const T& mu_Nb = oldGridN[1];
			const T& phi_del = oldGridN[2];
			const T& phi_lav = oldGridN[3];
			const T phi_gam = 1. - phi_del - phi_lav;

			vector<T> grad_phi_del = gradient(oldGrid, x, 2);
			vector<T> grad_phi_lav = gradient(oldGrid, x, 3);

			// compute grand potentials (Eqn. 19)
			const T w_gam = - (mu_Cr * mu_Cr * gam_X_CrCr)/2. - (mu_Nb * mu_Nb * gam_X_NbNb) / 2. - (mu_Cr * mu_Nb * gam_X_CrNb) - mu_Cr * xe_Cr_gam - mu_Nb * xe_Nb_gam;
			const T w_del = - (mu_Cr * mu_Cr * del_X_CrCr)/2. - (mu_Nb * mu_Nb * del_X_NbNb) / 2. - (mu_Cr * mu_Nb * del_X_CrNb) - mu_Cr * xe_Cr_del - mu_Nb * xe_Nb_del;
			const T w_lav = - (mu_Cr * mu_Cr * lav_X_CrCr)/2. - (mu_Nb * mu_Nb * lav_X_NbNb) / 2. - (mu_Cr * mu_Nb * lav_X_CrNb) - mu_Cr * xe_Cr_lav - mu_Nb * xe_Nb_lav;

			double myenergy = 0.;
			myenergy += 0.5 * kappa * grad_phi_del * grad_phi_del + omega * fdw(phi_del) + alpha * phi_del*phi_del * phi_lav*phi_lav;
			myenergy += 0.5 * kappa * grad_phi_lav * grad_phi_lav + omega * fdw(phi_lav) + alpha * phi_del*phi_del * phi_lav*phi_lav;
			myenergy += g(phi_gam) * w_gam + g(phi_del) * w_del + g(phi_lav) * w_lav;

			myenergy *= dV;

			#pragma omp atomic
			energy += myenergy;
		}

		if (rank==0)
			fprintf(efile, "%f,%f\n", elapsed, energy);

	} // for step in steps

	if (rank==0)
		fclose(efile);
}

} // namespace MMSP

template<typename T>
void computeCompositions(MMSP::vector<T>& v)
{
	const T& mu_Cr = v[0];
	const T& mu_Nb = v[1];

	const T& phi_del = v[2];
	const T& phi_lav = v[3];

	const T g_del = g(phi_del);
	const T g_lav = g(phi_lav);
	const T g_gam = 1. - g_del - g_lav;

	// gamma composition
	v[4] = xe_Cr_gam + gam_X_CrCr * mu_Cr + gam_X_CrNb * mu_Nb;
	v[5] = xe_Nb_gam + gam_X_NbCr * mu_Cr + gam_X_NbNb * mu_Nb;

	// delta composition
	v[6] = xe_Cr_del + del_X_CrCr * mu_Cr + del_X_CrNb * mu_Nb;
	v[7] = xe_Nb_del + del_X_NbCr * mu_Cr + del_X_NbNb * mu_Nb;

	// Laves composition
	v[8] = xe_Cr_lav + lav_X_CrCr * mu_Cr + lav_X_CrNb * mu_Nb;
	v[9] = xe_Nb_lav + lav_X_NbCr * mu_Cr + lav_X_NbNb * mu_Nb;

	// composition
	v[10] = g_gam * v[4] + g_del * v[6] + g_lav * v[8];
	v[11] = g_gam * v[5] + g_del * v[7] + g_lav * v[9];
}

template <int dim, typename T> MMSP::vector<T> ranged_laplacian(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, int start, int end)
{
	MMSP::vector<T> laplacian(end-start, 0.);
	const MMSP::vector<T>& yc = GRID(x);

	for (int d=0; d<dim; d++) {
		MMSP::vector<int> s(x);
		s[d] -= 1;
		const MMSP::vector<T>& yl = GRID(s);
		s[d] += 2;
		const MMSP::vector<T>& yh = GRID(s);
		s[d] -= 1;

		const T weight = 1.0 / (dx(GRID, d) * dx(GRID, d));

		for (int i = start; i<end; i++)
			laplacian[i-start] += weight * (yh[i] - 2. * yc[i] + yl[i]);
	}

	return laplacian;
}

template <int dim, typename T> MMSP::vector<T> divGradCr(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, int N)
{
	MMSP::vector<T> laplacian(N, 0.);
	const MMSP::vector<T>& yc = GRID(x);
	T Ml, Mc, Mh; // mobilities

	for (int d=0; d<dim; d++) {
		MMSP::vector<int> s(x);
		s[d] -= 1;
		const MMSP::vector<T>& yl = GRID(s);
		s[d] += 2;
		const MMSP::vector<T>& yh = GRID(s);
		s[d] -= 1;

		const T weight = 1.0 / (dx(GRID, d) * dx(GRID, d));

		for (int i=0; i<N; i++) {
			Mc =  D_Cr[i] * (g(yc[2]) * del_X_Cr[i] + g(yc[3]) * lav_X_Cr[i] + (1.-g(yc[2])-g(yc[3])) * gam_X_Cr[i]);
			Ml =  D_Cr[i] * (g(yl[2]) * del_X_Cr[i] + g(yl[3]) * lav_X_Cr[i] + (1.-g(yl[2])-g(yl[3])) * gam_X_Cr[i]);
			Mh =  D_Cr[i] * (g(yh[2]) * del_X_Cr[i] + g(yh[3]) * lav_X_Cr[i] + (1.-g(yh[2])-g(yh[3])) * gam_X_Cr[i]);

			laplacian[i] += weight * (0.5 * (Mh + Mc) * (yh[i] - yc[i]) - 0.5 * (Mc + Ml) * (yc[i] - yl[i]));
		}
	}

	return laplacian;
}

template <int dim, typename T> MMSP::vector<T> divGradNb(const MMSP::grid<dim,MMSP::vector<T> >& GRID, const MMSP::vector<int>& x, int N)
{
	MMSP::vector<T> laplacian(N, 0.);
	const MMSP::vector<T>& yc = GRID(x);
	T Ml, Mc, Mh; // mobilities

	for (int d=0; d<dim; d++) {
		MMSP::vector<int> s(x);
		s[d] -= 1;
		const MMSP::vector<T>& yl = GRID(s);
		s[d] += 2;
		const MMSP::vector<T>& yh = GRID(s);
		s[d] -= 1;

		const T weight = 1.0 / (dx(GRID, d) * dx(GRID, d));

		for (int i=0; i<N; i++) {
			Mc =  D_Nb[i] * (g(yc[2]) * del_X_Nb[i] + g(yc[3]) * lav_X_Nb[i] + (1.-g(yc[2])-g(yc[3])) * gam_X_Nb[i]);
			Ml =  D_Nb[i] * (g(yl[2]) * del_X_Nb[i] + g(yl[3]) * lav_X_Nb[i] + (1.-g(yl[2])-g(yl[3])) * gam_X_Nb[i]);
			Mh =  D_Nb[i] * (g(yh[2]) * del_X_Nb[i] + g(yh[3]) * lav_X_Nb[i] + (1.-g(yh[2])-g(yh[3])) * gam_X_Nb[i]);

			laplacian[i] += weight * (0.5 * (Mh + Mc) * (yh[i] - yc[i]) - 0.5 * (Mc + Ml) * (yc[i] - yl[i]));
		}
	}

	return laplacian;
}

#include "MMSP.main.hpp"
