import numpy as np, argparse, warnings, sys, os, h5py
from scipy.integrate import cumtrapz
from scipy.ndimage import map_coordinates
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
# The spatial information
parser.add_argument("-n",     type=int,   default=1000, help="Radial samples to output")
parser.add_argument("--rsub", type=int,   default=1,    help="Radial subsamples per sample")
parser.add_argument("-R",     type=float, default=10,   help="Max radius")
parser.add_argument("--ncut", type=int,   default=None, help="Only output first ncut points")
# Mass distribution
parser.add_argument("-r",     type=str, default=None,    help="Radii of features")
parser.add_argument("-w",     type=str, default=None,    help="Widths of features")
parser.add_argument("-A",     type=str, default=None,    help="Amplitudes of features")
parser.add_argument("-P", "--profile", type=str, default="gauss", help="Profiles of features")
parser.add_argument("-f",     type=str, default=None,    help="Read mass distribution from file.")
parser.add_argument("-T", "--type", type=str, default="dust", help="Type of energy: dust or light")
# Boundary condition
parser.add_argument("-B", "--boundary", type=str, default="none", help="Boundary condition: none or hawk")
# Time steps
parser.add_argument("-N", type=int, default=0, help="Number of time steps. 0 for snapshot")
parser.add_argument("--dt", type=float, default=0.1, help="Size of each time step")
parser.add_argument("--tsub", type=int, default=1, help="Time subsamples per timestep")
parser.add_argument("--tfull", type=int, default=0, help="Dump full frame every tfull samples")
# Misc
parser.add_argument("-I", "--integrator", type=str, default="trap", help="Integrator type: trap")
parser.add_argument("-i", type=int, default=0, help="Label output files starting from i")
parser.add_argument("-o", "--order", type=int, default=1, help="Interpolation order")
parser.add_argument("odir", nargs="?")
args = parser.parse_args()

# Natural units for this problem:
# r in units of lP, the planck length
# t in units of tP, the planck time
# This gives evaporation formula
#  R = (t/640pi)**(1/3)

def pad(a,n): return np.concatenate([a[:n],np.tile(a[-1],n-len(a[:n]))])
def gauss(x,x0,dx,A): return np.exp(-0.5*((x-x0)/dx)**2)*(A/(2*np.pi*dx**2)**0.5)
def flat(x,x0,dx,A): return (x>x0-dx/2)*(x<x0+dx/2)*(A/dx)
funcs = {"gauss":gauss, "flat":flat}
def isum(a, dx): return np.cumsum(a)*dx
def dsum(a, dx): return np.concatenate([a[:1],0.5*(a[2:]-a[:-2]),[a[-1]-a[-2]]])/dx
def itrap(a, dx): return cumtrapz(a,dx=dx,initial=0)
# Not the inverse of itrap, should fix
def dtrap(a, dx): return dsum(a,dx)
ints = {"sum":isum, "trap":itrap}
devs = {"sum":dsum, "trap":dtrap}
ifun = ints[args.integrator]
dfun = devs[args.integrator]

m  = args.rsub
n  = args.n*m
r  = np.linspace(1e-10, args.R, n)
dt = args.dt
dR = np.zeros(n)

if args.r:
	rs = np.array([float(w) for w in args.r.split(",")])
	ws = np.array([float(w) for w in args.w.split(",")])
	As = np.array([float(w) for w in args.A.split(",")])
	Ps = args.profile.split(",")
	N  = len(rs)
	ws, As = pad(ws,N), pad(As,N)
	Ps = Ps[:N]+Ps[:N][-1:]*(N-len(Ps[:N]))
	for ri, wi, Ai, Pi in zip(rs,ws,As,Ps):
		dR += funcs[Pi](r,ri,wi,Ai)
dr = r[1]
R = ifun(dR,dr)
if args.f:
	with h5py.File(args.f, "r") as hfile:
		R = hfile["data"].value[2]

def calc_metric(R):
	dR = dfun(R, dr)
	b = 1/(1-R/r)
	core = b*(1+dR if args.type == "light" else 1)
	a = np.exp(-ifun((1/r*(core-1))[::-1],dr)[::-1])
	# complete integral to infinity
	a = a/a[-1]*(1-R[-1]/r[-1])
	return a, b, dR

def boundary_none(t, r, rmax, R, a, b): return R
def smooth_evap(x, x0): return (x0**(5./3)*x**-2+x**-(1./3))**-1
def boundary_hawk(t, r, rmax, R, a, b):
	# Use the time explicitly to build the intensity.
	# R(r,t) = max(A*(r-vt),0)**(1./3)
	# But we want a smooth ramp-up to avoid ringing. So instead of (At)**1/3,
	# we can use A**(1/3) * (t**-2+t**-1/3)**-1
	R = R.copy()
	mask = r>rmax
	A  = (640*np.pi)**-(1./3)
	v = -(a[-1]/b[-1])**0.5
	t0 = 0.001
	R[mask] = A*smooth_evap(r[mask]-(rmax+v*t), t0)
	return R

boundary = {"none": boundary_none, "hawk":boundary_hawk}[args.boundary]

def step(t, R, a, b, dt):
	di = (a/b)**0.5*dt/dr
	i0 = np.arange(len(R))
	i  = i0+di
	if dt >= 0:
		ilast = np.where(i<=len(R)-1)[0][-1]
		R[:ilast] = map_coordinates(R, i[None,:ilast], mode="constant", order=args.order)
		R = boundary(t, i*dr, ilast*dr, R, a, b)
	else:
		# Boundary not supported here. This numerical solution is sadly not
		# time-reversible, and the reverse solution ends up proceeding significantly
		# faster than the forwards solution (by about 4%, though it's not a constant
		# ratio). A better integration scheme is needed to do this properly - forward
		# euler only takes you so far.
		R = map_coordinates(R, i[None], mode="constant", order=args.order)
	return R

def dump(fname, cols):
	np.savetxt(fname, np.array(cols).T[m/2::m][:args.ncut], fmt="%15.7e")
def dump_full(fname, cols):
	with h5py.File(fname, "w") as hfile:
		hfile["data"] = np.array(cols)

def fmt(fmt, i):
	try: return fmt % i
	except TypeError: return fmt

if args.N < 1:
	a, b, dR = calc_metric(R)
	dump("/dev/stdout", [r,dR,R,a,b])
else:
	try: os.mkdir(args.odir)
	except OSError: pass
	with open(args.odir+"/args.txt", "w") as f: print >> f, sys.argv
	i = args.i
	t = args.i*dt
	while i != args.N:
		print >> sys.stdout, "%5d" % i
		nb = args.tsub
		for bi in xrange(nb):
			a, b, dR = calc_metric(R)
			if bi == 0:
				dump(fmt(args.odir + "/step%06d.txt", i), [r,dR,R,a,b])
				if args.tfull and i % args.tfull == 0 and i != args.i:
					dump_full(fmt(args.odir + "/full%06d.hdf", i), [r,dR,R,a,b])
			R = step(t, R, a, b, dt/nb)
			t += dt/nb
		i += 1
