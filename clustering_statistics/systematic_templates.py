
import numpy as np

import lsstypes as types


def include_systematic_templates(window: types.WindowMatrix, templates: dict, effects: tuple | list=('auw',)):
    """Include systematic templates in the window matrix."""
    observable = window.observable
    windows, theories, _types = [], [], []
    for effect in effects:
        if effect == 'auw':
            diff = templates['auw'].match(observable).value() - templates['raw'].match(observable).value()
            # - sign, as applied on the theory side
            windows.append(- diff[:, None])
            theories.append(types.ObservableLeaf(value=np.array([1.]), scale=np.array([0.3])))
            _types.append(effect)
        elif effect == 'photo':
            diff_amr = templates['mock_wsys'].match(observable).value() - templates['data_nowsys'].match(observable).value()
            diff_wsys = templates['data_wsys'].match(observable).value() - templates['data_nowsys'].match(observable).value()
            diff_wsys -= diff_amr
            windows.append(- diff_amr[:, None])
            theories.append(types.ObservableLeaf(value=np.array([1.]), scale=np.array([0.01])))
            _types.append('amr')
            windows.append(- diff_wsys[:, None])
            theories.append(types.ObservableLeaf(value=np.array([0.]), scale=np.array([0.3])))
            _types.append('sys')
    theory = window.theory
    theory = types.ObservableTree([theory] + theories, types=['theory'] + _types)
    window = window.clone(value=np.concatenate([window.value()] + windows, axis=1), theory=theory)
    return window