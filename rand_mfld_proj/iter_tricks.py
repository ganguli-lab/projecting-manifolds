# -*- coding: utf-8 -*-
# =============================================================================
# Created on Wed Aug  3 14:45:37 2016
#
# @author: Subhy
#
# Module: iter_tricks
# =============================================================================
"""from sl_py_tools.display_tricks and sl_py_tools.iter_tricks

Tools for displaying temporary messages.

DisplayTemporary : class
    Class for temporarily displaying a message.
dcontext
    Display message during context.

Iterators for displaying progress.

DisplayCount : class
    Iterator for displaying loop counters.
DisplayBatch : class
    Iterator for displaying loop counters, returning slices.
DisplayEnumerate : class
    Like `zenumerate`, but using a `DisplayCount`.

For convenience:

denumerate : function
    Creates a `DisplayEnumerate`.
dbatch
    Creates a `DisplayBatch`.
rdcount, rdbatch, rdenumerate, rdzip
    Reversed versions of `dcount`, `dbatch`, `rdenumerate`, `dzip`.
You can set `start` and `stop` in `denumerate`, etc, but only via keyword
arguments.

.. warning:: Doesn't display properly on ``qtconsole``, and hence ``Spyder``.

Examples
--------
>>> with dcontext('running...'):
>>>     execute_fn(param1, param2)

>>> words = [''] * 4
>>> letters = 'xyz'
>>> counts = [1, 7, 13]
>>> for idx, key, num in denumerate('idx', letters, counts):
>>>     words[idx] = key * num
>>>     time.sleep(0.1)
>>> print(words)

>>> x = np.random.rand(1000, 3, 3)
>>> y = np.empty((1000, 3), dtype = complex)
>>> for s in dbatch('s', 0, len(x), 10):
>>>     y[s] = np.linalg.eigvals(x[s])
>>> print(x[15], y[15])
"""
__all__ = [
    'DisplayCount',
    'DisplayEnumerate',
    'DisplayBatch',
    'DisplayTemporary',
    'dcontext',
    'dcount',
    'dbatch',
    'denumerate',
    'rdbatch',
    'rdenumerate',
    ]
from functools import wraps
# All of these imports could be removed:
from abc import abstractmethod
from collections.abc import Iterator, Sized
from typing import ClassVar, Any, Optional, Union
from typing import Callable, Iterable, Tuple, Dict
from contextlib import contextmanager
import itertools
import io
import sys

assert sys.version_info[:2] >= (3, 6)

# =============================================================================
# %%* Class
# =============================================================================


class _DisplayState():
    """Internal state of a DisplayTemporary"""
    numchar: int
    nest_level: Optional[int]
    name: str

    def __init__(self, prev_state: Optional['_DisplayState'] = None):
        """Construct internal state"""
        self.nest_level = None
        self.numchar = 0
        self.name = "DisplayTemporary({})"
        if prev_state is not None:
            self.numchar = prev_state.numchar
            self.nest_level = prev_state.nest_level

    def format(self, *args, **kwds):
        """Replace field(s) in name"""
        self.name = self.name.format(*args, **kwds)

    def rename(self, name: str):
        """Replace prefix in name"""
        self.name = name + "({})"


class DisplayTemporary():
    """Class for temporarily displaying a message.

    Message erases when `end()` is called, or object is deleted.

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Attributes
    ----------
    output : bool, default : True
        Class attribute. Set it to `False` to suppress display.
    file : Optional[io.TextIOBase], default : None
        Class attribute. Output printed to `file`. If None, use `sys.stdout`.
    debug : bool, default : False
        Class attribute. Set it to `True` to check nesting.

    Class method
    ------------
    show(msg: str) -> DisplayTemporary:
        display `msg` and return class instance (needed to erase message).

    Methods
    -------
    begin(msg: str)
        for initial display of `msg`.
    update(msg: str)
        to erase previous message and display `msg`.
    end()
        to erase display.

    Example
    -------
    >>> dtmp = DisplayTemporary.show('running...')
    >>> execute_fn(param1, param2)
    >>> dtmp.end()
    """
    _state: _DisplayState

    # set output to False to suppress display
    output: ClassVar[bool] = True
    # write output to file. If None, use sys.stdout
    file: ClassVar[Optional[io.TextIOBase]] = None
    # set debug to True to check that displays are properly nested
    debug: ClassVar[bool] = False
    # used for debug
    _nactive: ClassVar[int] = 0

    def __init__(self, **kwds):
        self._state = _DisplayState(**kwds)

    def __del__(self):
        """Clean up, if necessary, upon deletion."""
        if self._state.numchar:
            self.end()

    def begin(self, msg: str = ''):
        """Display message.

        Parameters
        ----------
        msg : str
            message to display
        """
        if self._state.numchar:
            raise AttributeError('begin() called more than once.')
        self._state.format(msg)
        self._state.numchar = len(msg) + 1
        self._print(' ' + msg)
#        self._state['clean'] = False
        if self.debug:
            self._nactive += 1
            self._state.nest_level = self._nactive
            self._check()

    def update(self, msg: str = ''):
        """Erase previous message and display new message.

        Parameters
        ----------
        msg : str
            message to display
        """
        self._bksp(self._state.numchar)
        self._state.numchar = len(msg) + 1
        self._print(' ' + msg)
        if self.debug:
            self._check()

    def end(self):
        """Erase message.
        """
        self._erase(self._state.numchar)
        self._state.numchar = 0
        if self.debug:
            self._nactive -= 1

    def _print(self, text: str):
        """Print with customisations: same line and immediate output

        Parameters
        ----------
        text : str
            string to display
        """
        if self.output:
            print(text, end='', flush=True, file=self.file)

    def _bksp(self, num: int = 1, bkc: str = '\b'):
        """Go back num characters
        """
        if self.file is None:  # self.file.isatty() or self.file is sys.stdout
            pass
        elif self.file.seekable():
            self.file.seek(self.file.tell() - num)
            return

        # hack for jupyter's problem with multiple backspaces
        for i in bkc * num:
            self._print(i)
        # self._print('\b' * num)

    def _erase(self, num: int = 1):
        """Go back num characters
        """
        self._bksp(num)
        self._bksp(num, ' ')
        self._bksp(num)

    def _check(self):
        """Ensure that DisplayTemporaries are properly used
        """
        # raise error if ctr_dsp's are nested incorrectly
        if self._state.nest_level != self._nactive:
            msg1 = 'used at level {} '.format(self._nactive)
            msg2 = 'instead of level {}.'.format(self._state.nest_level)
            raise IndexError(self._state.name + msg1 + msg2)

    def rename(self, name):
        """Change name in debug message"""
        self._state.rename(name)

    @classmethod
    def show(cls, msg: str) -> 'DisplayTemporary':
        """Show message and return class instance.
        Parameters
        ----------
        msg : str
            message to display

        Returns
        -------
        disp_temp : DisplayTemporary
            instance of `DisplayTemporary`. Call `disp_temp.end()` or
            `del disp_temp` to erase displayed message.
        """
        disp_temp = cls()
        disp_temp.begin(msg)
        return disp_temp


# =============================================================================
# %%* Convenience functions
# =============================================================================


def _extract_name(args: Tuple[Any],
                  kwds: Dict[str, Any]) -> (Optional[str], Tuple[Any]):
    """Extract name from other args

    If name is in kwds, assume all of args is others, pop name from kwds.
    Else, if args[0] is a str or None, assume it's name & args[1:] is others.
    Else, name is None and all of args is others.
    """
    name = None
    others = args
    if 'name' not in kwds and isinstance(args[0], (str, type(None))):
        name = args[0]
        others = args[1:]
    name = kwds.pop('name', name)
    return name, others


def _extract_slice(args: Tuple[Optional[int], ...],
                   kwargs: Dict[str, Any]) -> Tuple[Optional[int], ...]:
    """Extract slice indices from args/kwargs

    Returns
    ----------
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.
    Or use keyword arguments.
    """
    inds = slice(*args)
    start = kwargs.pop('start', inds.start)
    stop = kwargs.pop('stop', inds.stop)
    step = kwargs.pop('step', inds.step)
    if start is None:
        start = 0
    if step is None:
        step = 1
    return start, stop, step


def _and_reverse(it_func: Callable):
    """Wrap iterator factory with reversed
    """
    @wraps(it_func)
    def rev_it_func(*args, **kwds):
        return reversed(it_func(*args, **kwds))

    new_name = it_func.__name__ + '.rev'
    rev_it_func.__name__ = new_name
    it_func.rev = rev_it_func
#    __all__.append(new_name)
#    setattr(current_module, new_name, rev_it_func)
    return it_func


# =============================================================================
# %%* Mixins for defining displaying iterators
# =============================================================================


class _DisplayCntState(_DisplayState):
    """Internal stae of a DisplayCount, etc."""
    prefix: Union[str, DisplayTemporary]
    formatter: str

    def __init__(self, prev_state: Optional[_DisplayState] = None):
        """Construct internal state"""
        super().__init__(prev_state)
        self.prefix = ' '
        self.formatter = '{:d}'

    def begin(self):
        """Display prefix"""
        self.format(self.prefix)
        self.prefix = DisplayTemporary.show(self.prefix)

    def end(self):
        """Display prefix"""
        self.prefix.end()


class _DisplayMixin(DisplayTemporary, Iterator):
    """Mixin providing non-iterator machinery for DisplayCount etc.

    This is an ABC. Only implements `begin`, `disp`, `end` and private stuff.
    Subclasses must implement `iter` and `next`.
    """
    counter: Optional[int]
    offset: int
    _state: _DisplayCntState

    def __init__(self, **kwds):
        """Construct non-iterator machinery"""
        super().__init__(**kwds)
        self._state = _DisplayCntState(self._state)
        self.counter = None
        self.rename(type(self).__name__)

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def begin(self, msg: str = ''):
        """Display initial counter with prefix."""
        self._state.begin()
        super().begin(self._str(self.counter) + msg)

    def update(self, msg: str = ''):
        """Erase previous counter and display new one."""
        dsp = self._str(self.counter)
        super().update(dsp + msg)

    def end(self):
        """Erase previous counter and prefix."""
        super().end()
        self._state.end()

    def _str(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
#        return self._frmt.format(ctr)
        return self._state.formatter.format(*(n + self.offset for n in ctrs))


class _AddDisplayToIterables(Iterator):
    """Wraps iterator to display progress.

    This is an ABC. Only implements `begin`, `disp` and `end`, as well as
    __init__ and __reversed__. Subclasses must implement `iter` and `next`.

    Specify ``displayer`` in keyword arguments of class definition to customise
    display. No default, but ``DisplayCount`` is suggested.
    The constructor signature is ``displayer(name, self._min_len(), **kwds)``.
    """
    _iterables: Tuple[Iterable, ...]
    display: _DisplayMixin

    def __init_subclass__(cls, displayer, **kwargs):
        super().__init_subclass__(**kwargs)
        cls.displayer = displayer

    def __init__(self, *args: Tuple[Union[str, Iterable, None], ...],
                 **kwds):
        """Construct non-iterator machinery"""
        name, self._iterables = _extract_name(args, kwds)
        self.display = self.displayer(name, self._min_len(), **kwds)
        self.display.rename(type(self).__name__)

    def __reversed__(self):
        """Prepare to display fina; counter with prefix."""
        self.display = reversed(self.display)
        self._iterables = tuple(reversed(seq) for seq in self._iterables)
        return self

    @abstractmethod
    def __next__(self):
        pass

    @abstractmethod
    def __iter__(self):
        pass

    def _min_len(self) -> Optional[int]:
        """Length of shortest sequence.
        """
        return min((len(seq) for seq in
                    filter(lambda x: isinstance(x, Sized), self._iterables)),
                   default=None)

    def begin(self, *args, **kwds):
        """Display initial counter with prefix."""
        self.display.begin(*args, **kwds)

    def update(self, *args, **kwds):
        """Erase previous counter and display new one."""
        self.display.update(*args, **kwds)

    def end(self):
        """Erase previous counter and prefix."""
        self.display.end()


# =============================================================================
# %%* Displaying iterator classes
# =============================================================================


class DisplayCount(_DisplayMixin, Sized):
    """Iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Construction
    ------------
    DisplayCount(name: str, low: int=0, high: int, step: int=1)

    DisplayCount(name: str, low: int=0, high: int)

    DisplayCount(name: str, high: int)

    DisplayCount(low: int=0, high: int, step: int=1)

    DisplayCount(low: int=0, high: int)

    DisplayCount(high: int)

    DisplayCount(name: str)

    DisplayCount()

    Parameters
    ----------
    name : str or None, optional
        name of counter used for prefix.
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.
    Or use keyword arguments.

    Attributes
    ----------
    output : bool, default : True
        Class attribute. Set it to `False` to suppress display.
    debug : bool, default : False
        Class attribute. Set it to `True` to check counter range and nesting.
    counter : int or None
        Instance attribute. Current value of the counter.

    Methods
    -------
    None of the following methods are ordinarily needed:

    begin()
        to initialise counter and display.
    update()
        to display current counter.
    end()
        to erase display after loops.

    Examples
    --------
    Triple nested loops:

    >>> import time
    >>> from iter_tricks import DisplayCount
    >>> for i in DisplayCount('i', 5):
    >>>     for j in DisplayCount('j', 6):
    >>>         for k in DisplayCount('k', 7):
    >>>             time.sleep(0.1)
    >>> print('done')

    Using `zip` and omitting `high`::

    >>> for i in DisplayCount('i', 5):
    >>>     for j, k in zip(DisplayCount('j'), [1, 7, 13]):
    >>>         time.sleep(1)
    >>>     time.sleep(3)
    >>> print('done')

    Raises
    ------
    IndexError
        If `DisplayCount.debug` is `True` and an instance is called with a
        counter value that is out of range, or if instances are improperly
        nested, e.g. if an outer DisplayCount is updated before an inner one
        has finished.

    See Also
    --------
    denumerate, DisplayZip, itertools.count
    """
    start: int
    stop: Optional[int]
    step: int

    def __init__(self, *args: Tuple[Union[str, int, None], ...],
                 **kwargs):
        name, sliceargs = _extract_name(args, kwargs)
        self.start, self.stop, self.step = _extract_slice(sliceargs, kwargs)
        # offset for display of counter, default: 1 if start==0, 0 otherwise
        self.offset = kwargs.pop('offset', int(self.start == 0))

        super().__init__(**kwargs)

        if name:
            self._state.prefix += name + ':'

        if self.stop:
            self.stop = self.start + self.step * len(self)
            num_dig = str(len(str(self.stop)))
            self._state.formatter = '{:>' + num_dig + 'd}/'
            self._state.formatter += self._str(self.stop - self.offset)[:-1]
        self._state.formatter += ','

    def __iter__(self):
        """Display initial counter with prefix."""
        self.counter = self.start
        self.begin()
        self.counter -= self.step
        return self

    def __reversed__(self):
        """Prepare to display final counter with prefix.
        Calling iter and then next will count down.
        """
        if self.stop is None:
            raise ValueError('Must specify stop to reverse')
        self.start, self.stop = self.stop - self.step, self.start - self.step
        self.step *= -1
        self._state.prefix += '-'
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        self.counter += self.step
        if (self.stop is None) or self.step*(self.stop - self.counter) > 0:
            self.update()
            return self.counter
        self.end()
        raise StopIteration()

    def __len__(self):
        """Number of entries"""
        if self.stop is None:
            raise ValueError('Must specify stop to define len')
        return (self.stop - self.start) // self.step

    def _check(self):
        """Ensure that DisplayCount's are properly used"""
        super()._check()
        # raise error if ctr is outside range
        if self.counter > self.stop or self.counter < self.start:
            msg1 = ' has value {} '.format(self.counter)
            msg2 = 'when upper limit is {}.'.format(self.stop)
            raise IndexError(self._state.name + msg1 + msg2)


class DisplayBatch(DisplayCount):
    """Iterate over batches, with counter display

    Similar to `DisplayCount` object, except at each iteration it yields a
    `slice` covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8(/2), k:  7/10(/5),'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.
    Or use keyword arguments.

    Yields
    ------
    batch_slice
        `slice` object that starts at current counter and stops at the next
        value with step size 1.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s in dbatch('s', 0, len(x), 10):
    >>>     y[s] = np.linalg.eigvals(x[s])
    """
    def __init__(self, *args: Tuple[Union[str, int, None], ...],
                 **kwargs):
        super().__init__(*args, **kwargs)

        if self.stop is None:
            self._state.formatter = '{:d}-{:d}'
        else:
            num_dig = len(str(self.stop))
            frmt = '{:>' + str(num_dig) + 'd}'
            self._state.formatter = frmt + '-' + frmt + '/'
            self._state.formatter += frmt.format(self.stop)
        self._state.formatter += ','

    def _str(self, *ctrs: int) -> str:
        """String for display of counter, e.g.' 7/12,'."""
#        return self._frmt.format(ctr)
        return super()._str(*itertools.chain(*((n, n + abs(self.step) - 1)
                                               for n in ctrs)))

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        counter = super().__next__()
        return slice(counter, counter + abs(self.step))


class DisplayEnumerate(_AddDisplayToIterables, displayer=DisplayCount):
    """Wraps iterator to display progress.

    Like `zenumerate`, but using a `DisplayCount`.
    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    sequences : Tuple[Iterable]
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.
        Note: argument is unpacked.
    **kwds
        Passed to `DisplayCount`

    Examples
    --------
    >>> import time
    >>> from iter_tricks import DisplayEnumerate
    >>> keys = 'xyz'
    >>> values = [1, 7, 13]
    >>> assoc = {}
    >>> for key, val in DisplayEnumerate('idx', keys, values):
    >>>     assoc[key] = val
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    DisplayCount, enumerate, zip
    """
    _iterator: Iterator

    def __iter__(self):
        """Display initial counter with prefix."""
        self._iterator = zip(self.display, *self._iterables)
        return self

    def __next__(self):
        """Increment counter, erase previous counter and display new one."""
        try:
            output = next(self._iterator)
        except StopIteration:
            self.display = None
            raise
        else:
            return output


# =============================================================================
# %%* Functions
# =============================================================================


@contextmanager
def dcontext(msg: str):
    """Display message during context.

    Prints message before entering context and deletes after.

    Parameters
    ----------
    msg : str
        message to display

    Example
    -------
    >>> with dcontext('running...'):
    >>>     execute_fn(param1, param2)
    """
    dtmp = DisplayTemporary.show(msg)
    try:
        yield
    finally:
        dtmp.end()


# - only saves ink


@_and_reverse
def dcount(*args: Tuple[Union[str, int, None], ...],
           **kwargs)-> DisplayCount:
    """Produces iterator for displaying loop counters.

    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns loop counter in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Parameters
    ----------
    name : str or None, optional
        name of counter used for prefix.
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.

    Returns
    -------
    disp_counter : DisplayCount
        An iterator that displays & returns counter value.

    Examples
    --------
    Triple nested loops:

    >>> import time
    >>> from iter_tricks import dcount
    >>> for i in dcount('i', 5):
    >>>     for j in dcount('j', 6):
    >>>         for k in dcount('k', 7):
    >>>             time.sleep(0.1)
    >>> print('done')

    Using `zip` and omitting `high`::

    >>> for i in dcount('i', 5):
    >>>     for j, k in zip(dcount('j'), [1, 7, 13]):
    >>>         time.sleep(1)
    >>>     time.sleep(3)
    >>> print('done')

    Raises
    ------
    IndexError
        If `DisplayCount.debug` is `True` and an instance is called with a
        counter value that is out of range, or if instances are improperly
        nested, e.g. if an outer DisplayCount is used before an inner one
        has finished.

    See Also
    --------
    DisplayCount, denumerate, dzip,
    DisplayEnumerate, DisplayZip,
    itertools.count
    """
    return DisplayCount(*args, **kwargs)


@_and_reverse
def denumerate(*args: Tuple[Union[str, Iterable, None], ...],
               **kwds)-> DisplayEnumerate:
    """Like `zenumerate`, but using a `DisplayCount`.

    Reads maximum couter value from min length of Sized `sequences`.
    Prints loop counter (plus 1), updates in place, and deletes at end.
    Returns (loop counter, sequence members) in each loop iteration.
    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 6/8, k:  7/10,'
    The output of `next` is a `tuple`: (counter, iter0, iter1, ...)

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    sequences : Tuple[Iterable]
        Containers that can be used in a ``for`` loop, preferably `Sized`,
        i.e. ``len(sequence)`` works, e.g. tuple, list, np.ndarray.
        Note: argument is unpacked.
    **kwds
        Passed to `DisplayCount`

    Returns
    -------
    disp_enum : DisplayEnumerate
        An iterator that displays & returns counter value & `sequences` entries

    Examples
    --------
    >>> import time
    >>> import numpy as np
    >>> from iter_tricks import denumerate
    >>> words = np.array([' ' * 13] * 3)
    >>> letters = 'xyz'
    >>> counts = [1, 7, 13]
    >>> for idx, key, num in denumerate('idx', letters, counts):
    >>>     words[idx] = key * num
    >>>     time.sleep(0.1)
    >>> print('done')

    See Also
    --------
    DisplayEnumerate, dzip
    DisplayZip, DisplayCount,
    enumerate, zip
    """
    return DisplayEnumerate(*args, **kwds)


@_and_reverse
def dbatch(*args: Tuple[Union[str, int, None], ...],
           **kwargs) -> DisplayBatch:
    """Iterate over batches, with counter display

    Similar to `dcount`, except at each iteration it yields a
    `slice` covering that step.

    Nested loops display on one line and update correctly if the inner
    DisplayCount/DisplayZip ends before the outer one is updated.
    Displays look like:
        ' i: 3/5, j: 3-4/8k:  6-10/10,'

    .. warning:: Doesn't display properly on ``qtconsole``, and hence Spyder.
    Instead, use in a console connected to the same kernel:
    ``cd`` to the folder, then type: ``jupyter console --existing``, and run
    your code there.

    Parameters
    ----------
    name : Optional[str]
        name of counter used for prefix.
    start : int or None, optional, default=0
        initial counter value (inclusive).
    stop : int or None, optional, default=None
        value of counter at, or above which, the loop terminates (exclusive).
    step : int or None, optional, default=1
        increment of counter after each loop.

    `start`, `stop` and `step` behave like `slice` indices when omitted.
    To specify `start/step` without setting `stop`, set `stop` to `None`.
    To specify `step` without setting `start`, set `start` to 0 or `None`.
    Or use keyword arguments.


    Returns
    -------
    disp_counter : DisplayBatch
        An iterator that displays counter & returns:

        batch_slice:
            `slice` object that starts at current counter and stops at next.

    Example
    -------
    >>> import numpy as np
    >>> x = np.random.rand(1000, 3, 3)
    >>> y = np.empty((1000, 3), dtype = complex)
    >>> for s in dbatch('s', 0, len(x), 10):
    >>>     y[s] = np.linalg.eigvals(x[s])
    """
    return DisplayBatch(*args, **kwargs)


# =============================================================================
# %%* Reversed iterator factories
# =============================================================================
rdbatch = dbatch.rev
rdenumerate = denumerate.rev
