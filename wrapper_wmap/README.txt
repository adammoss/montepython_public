Work-around (all credit due to S. More) to get the WMAP likelihood to work (again).
In principle, you just follow the usual setup instructions with 'waf' as described
in the docs but since that procedure for some reason does not create the `pywlik.py`
module, we added a `setup.py` to `src/python/` which enforces the creation of that
module when invoked with `python setup.py install`.
This might bounce back though because some folders are not created automatically.
Just create these by hand and rerun.
Moreover, the `setup.py` might need to prepended with some system-specific compiler
flags (e.g. pointing to clapack.h and so on...).

Last but not least, we moved the default `wrapper_wmap` to `wrapper_wmap_v4p1`
because that is currently set up to install the v4.1 likelihood (i.e. the 7-year
release of WMAP) and included all necessary changes to install the most recent v5.0
likelihood (the 9-year release of WMAP) in the 'new' `wrapper_wmap` folder.

Thanks to F. Koehlinger and S. More
