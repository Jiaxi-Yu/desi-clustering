import os
from matplotlib import pyplot as plt
import numpy as np

import lsstypes as types
from pathlib import Path
from clustering_statistics import tools


def check_if_exists(fns, include_dubious=False):
    """
    Check which files exist, optionally including dubious realizations.
    Prioritizing good realizations over dubious ones
    """
    
    all_fns = []
    
    for fn in fns:
        # Always check for the 'good' file first
        if os.path.exists(fn):
            all_fns.append(fn)
        elif include_dubious:
            # Generate potential dubious filenames
            dubious_fn1 = str(fn).replace('mock', 'dubious_mock')
            dubious_fn2 = str(fn).replace('mock', 'dubious_mock_')
            
            # Include the first one that exists
            if os.path.exists(dubious_fn1):
                all_fns.append(dubious_fn1)
            elif os.path.exists(dubious_fn2):
                all_fns.append(dubious_fn2)
    
    if include_dubious:
        print('WARNING! Using measurements from dubious realizations.')
    
    return all_fns
    
def get_means_covs(kind, versions, tracer, zrange, region, stats_dir, project='', ells=(0,2,4), rebin=1, include_dubious=False):
    # Note: include_dubious = True does nothing if imock='*'
    kind_ = kind
    stats_, means, covs = {}, {}, {}
    for version in versions:
        use_theory  = versions[version].get('theory',False)
        use_dubious =  versions[version].get('include_dubious',False)
        tracer = versions[version].get('tracer',tracer) # try to use tracer provided in dictionary. Useful for when tracer names do not match, e.g., BGS_BRIGHT-21.35 vs BGS_BRIGHT.
        if use_theory:
            kind = 'theory_'+kind_
            if 'mesh3' in kind or 'particle2' in kind:
                raise NotImplementedError(f'kind {kind} is not supported.')
        else:
            kind = kind_
        kw = dict(tracer=tracer, kind=kind, stats_dir=stats_dir, project=project, zrange=zrange, region=region)
        for name in ['version', 'weight', 'cut', 'auw', 'extra']:
            kw[name] = versions[version][name]
        if 'ELG' in kw['tracer']:
            if 'complete' in kw['version']:
                kw['tracer'] = 'ELG_LOP'
            elif 'data' in kw['version']:
                kw['tracer'] = 'ELG_LOPnotqso'
        if 'mesh3' in kind:
            kw['basis'] = 'sugiyama-diagonal'
            kw['auw'] = False  
        
        if kw['version'] == 'data-dr2-v2':
            fns = tools.get_stats_fn(**kw)
        else:
            imocks = None if use_theory else versions[version]['imocks']
            if imocks is None:
                fns = tools.get_stats_fn(**kw, imock='*')
            else:
                fns = [tools.get_stats_fn(**kw, imock=imock) for imock in imocks]
                # fns = [fn for fn in fns if os.path.exists(fn)]
        if isinstance(fns, (str, Path)):
            fns = [fns]
        fns = check_if_exists(fns, include_dubious=use_dubious)
        
        stats = [types.read(fn) for fn in fns]
        if 'particle2_correlation' in kind:
            stats = [stat.project(ells=ells) for stat in stats]
            stats_[version] = stats
            means[version]  = types.mean(stats).select(s=slice(0, None, rebin))
        else:
            stats_[version] = stats
            means[version]  = stats[0] if use_theory else types.mean(stats)
            means[version]  = means[version].select(k=slice(0, None, rebin))
        if len(stats) > 1:
            covs[version] = types.cov(stats).at.observable.match(means[version])
        else:
            if use_theory:
                covs[version] = types.read(str(fns[0]).replace('theory','covariance')).at.observable.get(observables='spectrum2')
                # covs[version] = covs[version].at.observable.match(means[version])
            else:
                covs[version] = None
    return stats_, means, covs


def plot_stats(kind, versions, tracer, zrange, region, stats_dir, project='', include_dubious=False, ells=(0,2,4), rebin=1, reference=None, plot_all=False, imocks=None, ylim=(-1.5, 1.5), figure=None, ax_col=0, linestyles=None, lw=2, colors=None, scaling='kpk', save_fn=None, title=None, legend_ncol=1, legend_title=''):
    if reference is None:
        # use first item from versions as reference
        reference = next(iter(versions))
    if linestyles is None: linestyles = dict(zip(versions, ['-']*len(versions)))
    if colors is None: colors = dict(zip(versions, [f'C{i:d}' for i in range(len(versions))]))
    if figure is None:
        fig, lax = plt.subplots(len(ells) * 2, figsize=(6, 10), sharex=True, gridspec_kw={'height_ratios': [2.5, 1] * len(ells)})
    else:
        fig, axes = figure
        if axes.ndim == 1:
            lax = axes
        else:
            lax = axes[:, ax_col]
    k_exp = 1 if scaling == 'kpk' else 0
    s_exp = 2
    if 'mesh2_spectrum' in kind:
        stats, means, covs = get_means_covs(kind, versions, tracer, zrange, region, stats_dir, project=project, rebin=rebin,  include_dubious=include_dubious)
        versions = list(means)
        if title is None:
            lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        else:
            lax[0].set_title(title)
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            if scaling == 'kpk':
                ax.set_ylabel(rf'$k P_{ell:d}(k)$ [$(\mathrm{{Mpc}}/h)^2$]')
            if scaling == 'loglog':
                ax.set_ylabel(rf'$P_{ell:d}(k)$ [$(\mathrm{{Mpc}}/h)^3$]')
                ax.set_yscale('log')
                ax.set_xscale('log')
            if plot_all:
                for iversion, version in enumerate(versions):
                    for stat in stats[version]:
                        pole = stat.get(ell)
                        value = pole.coords('k')**k_exp * pole.value().real
                        ax.plot(pole.coords('k'), value, color=colors[version], linestyle='-', lw=1, alpha=0.1)                
            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                value = pole.coords('k')**k_exp * pole.value().real
                if 'data' in version or version == reference:
                    # cov = covs[reference].copy().at.observable.match(means[version])
                    std = pole.coords('k')**k_exp * covs[reference].copy().at.observable.match(means[version]).at.observable.get(ell).std().real
                    ax.fill_between(pole.coords('k'), value - std, value + std, color=colors[version], alpha=0.2)
                ax.plot(pole.coords('k'), value, color=colors[version], linestyle=linestyles[version], label=version+f' (#{len(stats[version])})', lw=lw)
            if ill == 0: ax.legend(frameon=False, ncol=legend_ncol, title=legend_title)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta P_{ell:d} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole_reference = means[reference].get(ell).copy()
                pole = means[version].get(ell).copy()
                # solve issue with mismatch in k in theory and mocks
                if pole.size > pole_reference.size:
                    pole = pole.select(k=slice(1,None,1))
                elif pole.size < pole_reference.size:
                    pole_reference = pole_reference.select(k=slice(1,None,1))
                assert np.allclose(pole.coords('k'),pole_reference.coords('k')), f'Something went wrong when slicing k. {pole.coords('k')[-1],pole_reference.coords('k')[-1]}'
                std = covs[reference].copy().at.observable.match(means[version]).at.observable.get(ell).std().real
                # print(tracer,zrange,version,kind,ell,std.min())
                ax.plot(pole.coords('k'), (pole.value() - pole_reference.value()).real / std, color=colors[version], linestyle=linestyles[version], lw=lw)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    elif 'mesh3_spectrum_sugiyama-diagonal' in kind:
        stats, means, covs = get_means_covs('mesh3_spectrum', versions, tracer, zrange, region, stats_dir, project=project, rebin=rebin,  include_dubious=include_dubious)
        versions = list(means)
        if title is None:
            lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        else:
            lax[0].set_title(title)
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            if scaling == 'kpk':
                ax.set_ylabel(rf'$k^2 B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}(k, k)$ [$(\mathrm{{Mpc}}/h)^6$]')
            if scaling == 'loglog':
                ax.set_ylabel(rf'$B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}}(k, k)$ [$(\mathrm{{Mpc}}/h)^4$]')
                ax.set_yscale('log')
                ax.set_xscale('log')
            if plot_all:
                for iversion, version in enumerate(versions):
                    for stat in stats[version]:
                        pole = stat.get(ell)
                        x = pole.coords('k')[..., 0]
                        value = (x**2)**k_exp * pole.value().real
                        ax.plot(x, value, color=colors[version], linestyle='-', lw=1, alpha=0.1)
            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                x = pole.coords('k')[..., 0]
                value = (x**2)**k_exp * pole.value().real
                if 'data' in version or version == reference:
                    std = (x**2)**k_exp * covs[reference].at.observable.get(ell).std().real
                    ax.fill_between(x, value - std, value + std, color=colors[version], alpha=0.2)
                ax.plot(x, value, color=colors[version], linestyle=linestyles[version], label=version+f' (#{len(stats[version])})', lw=lw)
                if ill == 0: ax.legend(frameon=False, ncol=legend_ncol, title=legend_title)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta B_{{{ell[0]:d}{ell[1]:d}{ell[2]:d}}} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                std = covs[reference].at.observable.get(ell).std().real
                # print(tracer,zrange,version,kind,ell,std.min())
                x = pole.coords('k')[..., 0]
                ax.plot(x, (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version], lw=lw)
        lax[-1].set_xlabel(r'$k$ [$h/\mathrm{Mpc}$]')

    elif 'particle2_correlation' in kind:
        stats, means, covs = get_means_covs(kind, versions, tracer, zrange, region, stats_dir, project=project, ells=ells, rebin=rebin,  include_dubious=include_dubious)
        versions = list(means)
        if title is None:
            lax[0].set_title(f'{tracer} in {region} {zrange[0]:.1f} < z < {zrange[1]:.1f}')
        else:
            lax[0].set_title(title)
        for ill, ell in enumerate(ells):
            ax = lax[2 * ill]
            ax.set_ylabel(rf'$s^2 \xi_{ell:d}(s)$ [$(\mathrm{{Mpc}}/h)^2$]')

            if plot_all:
                for iversion, version in enumerate(versions):
                    for stat in stats[version]:
                        pole = stat.get(ell)
                        value = pole.coords('s')**s_exp * pole.value().real
                        ax.plot(pole.coords('s'), value, color=colors[version], linestyle='-', lw=1, alpha=0.1)                 
            for iversion, version in enumerate(versions):
                if ell not in means[version].ells: continue
                pole = means[version].get(ell)
                value = pole.coords('s')**s_exp * pole.value().real
                if 'data' in version or version == reference:
                    std = pole.coords('s')**s_exp * covs[reference].at.observable.get(ell).std().real
                    ax.fill_between(pole.coords('s'), value - std, value + std, color=colors[version], alpha=0.2)
                ax.plot(pole.coords('s'), value, color=colors[version], linestyle=linestyles[version], label=version+f' (#{len(stats[version])})', lw=lw)
            if ill == 0: ax.legend(frameon=False, ncol=legend_ncol, title=legend_title)
            ax.grid(True)
            ax = lax[2 * ill + 1]
            ax.set_ylabel(rf'$\Delta P_{ell:d} / \sigma(k)$')
            ax.grid(True)
            ax.set_ylim(*ylim)
            for iversion, version in enumerate(versions):
                if 'data' in version or version == reference: continue
                pole = means[version].get(ell)
                # std = covs[reference].at.observable.get(ell).std().real
                std = covs[reference].at.observable.get(ell).std().real
                # print(tracer,zrange,version,kind,ell,std.min())
                ax.plot(pole.coords('s'), (pole.value() - means[reference].get(ell).value()).real / std, color=colors[version], linestyle=linestyles[version], lw=lw)
        lax[-1].set_xlabel(r'$s$ [$h^{-1}\mathrm{Mpc}$]')

    if save_fn and figure is None:
        plt.tight_layout()
        fig.savefig(save_fn, bbox_inches='tight', pad_inches=0.1, dpi=200)
        plt.show()

