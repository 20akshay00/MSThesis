from scipy.linalg import eigh_tridiagonal
import numpy as np
import argparse, h5py

def get_bloch_spectrum(V0, kl, q, l_max):
    l = np.arange(-l_max, l_max+1)
    E, psi = eigh_tridiagonal((2 * l + q/kl) ** 2 + V0/2, -V0/4 * np.ones(len(l) - 1))
    return np.real(E), psi.T

def to_pos_basis(phi_l, kl, l_max):
    l = np.arange(-l_max, l_max+1)
    return lambda x: (1/np.sqrt(2 * np.pi)) * np.sum(phi_l * np.exp(2j * kl * x * l))
    
def func_bloch(psi, q_ind, band_num, kl, l_max):
    return to_pos_basis(psi[q_ind, band_num], kl, l_max)

def func_wannier(bloch_fn, k, R):
    N = len(k)
    def func(x):
        bloch_vals = np.array([fn(x) for fn in bloch_fn])
        return 1/np.sqrt(N) * np.sum(bloch_vals * np.exp(1j * k * (x - R)), axis = 0)

    return func

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
    description ='''Calculates the bloch spectrum in the first Brillouin zone for a sinusoidal potential.''')
    parser.add_argument('V0', nargs='?', type=float, default=0, help='V0 of the V0 * sin^2(kx) potential')
    parser.add_argument('k', nargs='?', type=float, default=np.pi, help='k of the V0 * sin^2(kx) potential')
    parser.add_argument('l_max', nargs='?', type=int, default=10, help='Such that number of basis elements is (2 * l_max + 1)')
    parser.add_argument('num_q', nargs='?', type=int, default=100, help='Number of points in the Brillouin Zone')
    args = parser.parse_args()

    V0, kl, l_max, num_q = args.V0, args.k, args.l_max, args.num_q #potential depth, period

    q = np.linspace(-np.pi, np.pi, num_q)
    psi = np.empty((len(q), 2 * l_max + 1, 2 * l_max + 1), dtype = np.complex128)
    E = np.empty((len(q), 2 * l_max + 1))

    for i, k in enumerate(q):
        E[i], psi[i] = get_bloch_spectrum(V0, kl, k, l_max)

    with h5py.File(f'Bloch-V={V0}_k={np.round(kl, 3)}_num-basis={2 * l_max + 1}_num-q={num_q}.hdf5', 'w') as f:
        f.create_dataset("Parameters", data = [V0, kl, l_max, num_q])
        f.create_dataset("q", data = q)
        f.create_dataset("psi", data = psi)
        f.create_dataset("E", data = E)

    