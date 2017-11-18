# -*- coding: utf-8 -*-
"""
Created on Wed Aug  3 14:45:37 2016

@author: Subhy

Module: disp_counter
====================

class: disp_counter
    Iterator for displaying loop counters.
function: denum
    Like `enumerate` + `zip`, but using a `display_counter`.

Examples
========
    for i in display_counter('i', 5):
        for j in display_counter('j', 6):
            for k in display_counter('k', 4, 10):
                time.sleep(0.1)
    print('done')

    for i in display_counter('i', 5):
        for j, k in zip(display_counter('j', 8), [1, 7, 13]):
            time.sleep(0.5)
        time.sleep(1)
    print('done')

    s={}
    for i in display_counter('i', 5):
        for j, key, val in denum( 'j', ('x', 'y', 'z'), [1, 7, 13]):
            s[key] = val
        time.sleep(1)
    print('done')
"""

from collections.abc import Iterator, Sized


def denum(*args, **kwargs)-> Iterator:
    """Like `enumerate` + `zip`, but using a `display_counter`.

    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    display_counter ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.

    Parameters
    ----------
    name : str or None, optional
        name of counter used for prefix.
    sequences : Tuple(Iterable)
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.

    Examples
    --------
    >>> d = {}
    >>> for j, key, val in denum( 'j', ('x', 'y', 'z'), [1, 7, 13]):
    >>>     s[key] = val
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    display_counter
    """
    name = kwargs.get('name', None)
    sequences = kwargs.get('sequences', args)
    if 'name' not in kwargs and (isinstance(args[0], str) or args[0] is None):
        name = args[0]
        if 'sequences' not in kwargs:
            sequences = args[1:]

    max_cnt = None
    for sequence in sequences:
        if isinstance(sequence, Sized):
            if max_cnt is None:
                max_cnt = len(sequence)
            else:
                max_cnt = min(max_cnt, len(sequence))

    return zip(display_counter(name, max_cnt), *sequences)


class display_counter(Iterator, Sized):
    """Iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    display_counter ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.

    Construction
    ------------
    display_counter(name: str, low: int=0, high: int, step: int=1)

    display_counter(name: str, low: int=0, high: int)

    display_counter(low: int=0, high: int, step: int=1)

    display_counter(name: str, high: int)

    display_counter(low: int=0, high: int)

    display_counter(high: int)

    display_counter(name: str)

    display_counter()

    Parameters
    ----------
    name : str or None, optional
        name of counter used for prefix.
    low : int, optional
        optional initial counter value, default: 0.
    high : int or None, optional
        max value of counter + 1, at, or above which, the loop terminates.
    step : int, optional
        increment of counter for each loop, default: 1.

    If you want to choose `low` without choosing `high`, use ``high=None``.

    Attributes
    ----------
    output : bool
        Class attribute. Set it to `False` to suppress printing.
        Default: `True`.
    debug : bool
        Class attribute. Set it to `True` to check counter range and nesting.
        Default: `False`
    counter : int or None
        Instance attribute. Current value of the counter.

    Methods
    -------
    None of the following methods are necessary:

    start()
        to initialise and display.
    disp()
        to display current counter.
    end()
        to erase after loop.

    Examples
    --------
    Triple nested loops:

    >>> for i in display_counter('i', 5):
    >>>     for j in display_counter('j', 6):
    >>>         for k in display_counter('k', 7):
    >>>             time.sleep(0.1)
    >>> print('done')

    Using `zip` and omitting `high`::

    >>> for i in display_counter('i', 5):
    >>>     for j, k in zip(display_counter('j'), [1, 7, 13]):
    >>>         time.sleep(1)
    >>>     time.sleep(3)
    >>> print('done')

    Raises
    ------
    IndexError
        If `display_counter.debug` is `True` and an instance is called with a
        counter value that is out of range, or if instances are improperly
        nested, e.g. if an outer display_counter is used before an inner one
        has finished.

    See Also
    --------
    disp_enum
    """
    _nactive = 0
    # set debug to True to check that counters are in range and properly nested
    debug = False
    # set output to False to suppress counter display
    output = True

    def __init__(self, *args, **kwargs):
        self._prefix = ' '
        self._min = kwargs.get('low', 0)
        self._max = kwargs.get('high', None)
        self._step = kwargs.get('step', 1)

        # Deal with optional arguments
        if 'name' in kwargs:
            self._prefix += kwargs['name'] + ': '
            inds = slice(*args)
        elif len(args) > 0 and isinstance(args[0], str):
            self._prefix += args[0] + ': '
            inds = slice(*args[1:])
        else:
            inds = slice(*args)

        if 'low' not in kwargs and inds.start is not None:
            self._min = inds.start
        if 'high' not in kwargs and inds.stop is not None:
            self._max = inds.stop
        if 'step' not in kwargs and inds.step is not None:
            self._step = inds.step

        if self._max is not None:
            num_dig = len(str(self._max))
            self._frmt = '{:>' + str(num_dig) + 'd}'
            self._frmt += '/' + self._frmt.format(self._max) + ','
        else:
            self._frmt = '{:d},'
        self.counter = None
        self._nest_level = None

    def __iter__(self):
        """Display initial counter with prefix."""
        self.start()
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        self.counter += self._step
        if (self._max is not None) and self.counter >= self._max:
            self.end()
            raise StopIteration()
        else:
            self.disp()
            return self.counter

    def __del__(self):
        """Clean up, if necessary"""
        if (self._max is None) or ((self.counter is not None) and
                                   (self.counter < self._max)):
            self.end()

    def __len__(self):
        """Number of entries"""
        return (self._max - self._min) // self._step

    def start(self):
        """Display initial counter with prefix."""
        self.counter = self._min - self._step
        self._print(self._prefix + self._str(self._min))
        if self.debug:
            self._nactive += 1
            self._nest_level = self._nactive
            self._check()

    def disp(self):
        """Erase previous counter and display new one."""
        dsp = self._str(self.counter)

        if self._max is None:
            num_digits = len(self._str(self.counter - self._step))
        else:
            num_digits = len(dsp)

        self._print('\b' * num_digits)
        self._print(dsp)
        if self.debug:
            self._check()

    def end(self):
        """Erase previous counter and prefix."""
        num_dig = len(self._prefix) + len(self._str(self.counter - self._step))
        self._print('\b \b' * num_dig)
        if self.debug:
            self._nactive -= 1

    def _str(self, ctr: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
        return self._frmt.format(ctr + 1)

    def _print(self, text: str):
        """Print with customisations: same line and immediate output"""
        if self.output:
            print(text, end='', flush=True)

    def _check(self):
        """Ensure that ctr_dsp's are properly used"""
        # raise error if ctr is outside range
        if self.counter > self._max or self.counter < self._min:
            msg1 = 'ctr_dsp{}'.format(self.prefix)
            msg2 = 'has value {} '.format(self.counter)
            msg3 = 'when upper limit is {}.'.format(self._max)
            raise IndexError(msg1 + msg2 + msg3)
        # raise error if ctr_dsp's are nested incorrectly
        if self._nest_level != self._nactive:
            msg1 = 'ctr_dsp{}'.format(self.prefix)
            msg2 = 'used at level {} '.format(self._nactive)
            msg3 = 'instead of level {}.'.format(self._nest_level)
            raise IndexError(msg1 + msg2 + msg3)
