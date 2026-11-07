"""Transit light curve of a dark disc over a star with quadratic limb darkening."""

from velociraptor.analytic import quadratic_flux
from velociraptor.quadratic import (
    orbit_series,
    orbit_taylor,
    quadrature_flux,
    separation,
    transit_flux,
)

__all__ = ["orbit_series", "orbit_taylor", "quadratic_flux", "quadrature_flux", "separation", "transit_flux"]
