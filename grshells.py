import numpy as np, argparse, warnings, sys
from scipy.integrate import cumtrapz
from scipy.ndimage import map_coordinates
warnings.filterwarnings("ignore")
parser = argparse.ArgumentParser()
parser.add_argument("-n", type=int,   default=1000)
parser.add_argument("-m", type=int,   default=1)
parser.add_argument("-R", type=float, default=10)
parser.add_argument("-r", type=str, default="2")
parser.add_argument("-w", type=str, default="0.1")
parser.add_argument("-A", type=str, default="1")
parser.add_argument("-P", "--profile", type=str, default="gauss")
parser.add_argument("-T", "--type", type=str, default="dust")
parser.add_argument("-I", "--integrator", type=str, default="trap")
parser.add_argument("--dt", type=float, default=0.1)
parser.add_argument("-B", "--boundary", type=str, default="none")
parser.add_argument("-i", type=int, default=0)
parser.add_argument("-N", type=int, default=0)
parser.add_argument("-s", type=int, default=1)
parser.add_argument("--boost", type=int, default=0)
parser.add_argument("--boost-factor", type=int, default=100)
parser.add_argument("--interpolation-order", type=int, default=1)
parser.add_argument("--ncut",default=None,type=int)
parser.add_argument("ofmt", nargs="?")
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
def dsum(a, dx): return np.concatenate([a[:1],a[1:]-a[:-1]])/dx
def itrap(a, dx): return cumtrapz(a,dx=dx,initial=0)
# Not the inverse of itrap, should fix
def dtrap(a, dx): return dsum(a,dx)
ints = {"sum":isum, "trap":itrap}
devs = {"sum":dsum, "trap":dtrap}
ifun = ints[args.integrator]
dfun = devs[args.integrator]

m  = args.m
n  = args.n*m
r  = np.linspace(0, args.R, n)
rs = np.array([float(w) for w in args.r.split(",")])
ws = np.array([float(w) for w in args.w.split(",")])
As = np.array([float(w) for w in args.A.split(",")])
Ps = args.profile.split(",")
N  = len(rs)
ws, As = pad(ws,N), pad(As,N)
Ps = Ps[:N]+Ps[:N][-1:]*(N-len(Ps[:N]))

dt = args.dt
dR = np.zeros(n)
for ri, wi, Ai, Pi in zip(rs,ws,As,Ps):
	dR += funcs[Pi](r,ri,wi,Ai)
dr = r[1]
R = ifun(dR,dr)

def calc_metric(R):
	dR = dfun(R, dr)
	b = 1/(1-R/r)
	core = b*(1+dR if args.type == "light" else 1)
	a = np.exp(-ifun((1/r*(core-1))[::-1],dr)[::-1])
	# complete integral to infinity
	a = a/a[-1]*(1-R[-1]/r[-1])
	return a, b, dR

def boundary_none(t, r, rmax, R, a, b): return R
def boundary_hawk(t, r, rmax, R, a, b):
	# At rmax we have dR(rmax)/dt = Lhawk. Since the shells move at
	# dr/dt = -(a/b)**0.5, we have R(r+dr) = R(r) + dR/dr * dr
	# = R(r) + dR/dt * dt/dr * dr
	R = R.copy()
	mask = r>rmax
	R[mask] += 0.01*(b[mask]/a[mask])**0.5 * (r[mask]-rmax) / np.maximum(0.04,R[mask])**2
	return R
def boundary_hawk2(t, r, rmax, R, a, b):
	# At the boundary light moves at dr/dt = -(a/b)**0.5. The time-profile
	# is R,t propto 1/R^2 => R^3 propto t => R propto t^{1/3}
	# Far away a and b are both almost constant, so the shells move
	# at speed v=(a/b)**0.5
	# Hence R(r,t) = A max(0,r-vt)^{1/3}
	# If at r=r0 R(r0,t) = R0(t) we can normalize this
	# R(r0,t) = A (r0-vt)^{1/3} = R0(t) => vt=r0-(R0/A)^3
	# R(r,t) = (A^3*(r-r0)+R0^3)^{1/3}
	R = R.copy()
	mask = r>rmax
	A = 1
	R[mask] = (A**3*(r[mask]-rmax)+R[-1]**3)**(1.0/3)
	return R

def smooth_evap(x, x0): return (x0**(5./3)*x**-2+x**-(1./3))**-1
def boundary_hawk3(t, r, rmax, R, a, b):
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
	#R[mask] = np.maximum(0,(A*(r[mask]-(rmax+v*t)))**(1./3))
	return R

boundary = {"none": boundary_none, "hawk":boundary_hawk, "hawk2":boundary_hawk2, "hawk3":boundary_hawk3}[args.boundary]

def step(t, R, a, b, dt):
	i = np.arange(len(R))+(a/b)**0.5*dt/dr
	ilast = np.where(i<=len(R)-1)[0][-1]
	R[:ilast] = map_coordinates(R, i[None,:ilast], mode="constant", order=args.interpolation_order)
	R = boundary(t, i*dr, ilast*dr, R, a, b)
	return R

def dump(fname, cols):
	np.savetxt(fname, np.array(cols).T[m/2::m][:args.ncut], fmt="%15.7e")

def fmt(fmt, i):
	try: return fmt % i
	except TypeError: return fmt

if args.ofmt is None:
	a, b, dR = calc_metric(R)
	dump("/dev/stdout", [r,dR,R,a,b])
else:
	i = 0
	t = 0
	while True:
		print >> sys.stdout, "%5d" % i
		j = i-args.i
		nb = args.boost_factor if i < args.boost else 1
		for bi in xrange(nb):
			a, b, dR = calc_metric(R)
			if bi == 0 and j >= 0 and (j < args.N or args.N == 0) and j%args.s == 0:
				dump(fmt(args.ofmt, i), [r,dR,R,a,b])
			if j >= args.N: break
			R = step(t, R, a, b, dt/nb)
			t += dt/nb
		i += 1
