# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------
# Copyright (c) 2011-2012 Lambda Foundry, Inc. and PyData Development Team
# Copyright (c) 2013 Jev Kuznetsov and contributors
# Copyright (c) 2014-2015 Scott Hansen <firecat4153@gmail.com>
# Copyright (c) 2014-2016 Yuri D'Elia "wave++" <wavexx@thregr.org>
# Copyright (c) 2014- Spyder Project Contributors
#
# Components of gtabview originally distributed under the MIT (Expat) license.
# This file as a whole distributed under the terms of the New BSD License
# (BSD 3-clause; see NOTICE.txt in the Spyder root directory for details).
# -----------------------------------------------------------------------------

"""
Pandas DataFrame Editor Dialog.

DataFrameModel is based on the class ArrayModel from array editor
and the class DataFrameModel from the pandas project.
Present in pandas.sandbox.qtpandas in v0.13.1.

DataFrameHeaderModel and DataFrameLevelModel are based on the classes
Header4ExtModel and Level4ExtModel from the gtabview project.
DataFrameModel is based on the classes ExtDataModel and ExtFrameModel, and
DataFrameEditor is based on gtExtTableView from the same project.

DataFrameModel originally based on pandas/sandbox/qtpandas.py of the
`pandas project <https://github.com/pandas-dev/pandas>`_.
The current version is qtpandas/models/DataFrameModel.py of the
`QtPandas project <https://github.com/draperjames/qtpandas>`_.

Components of gtabview from gtabview/viewer.py and gtabview/models.py of the
`gtabview project <https://github.com/TabViewer/gtabview>`_.
"""

# Standard library imports
import time
import math
import logging
import sys

logger = logging.getLogger("Test")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
ch.setFormatter(formatter)
logger.addHandler(ch)
# logger.setLevel(logging.INFO)
# handler = logging.FileHandler('test.log')
# formatter = logging.Formatter('%(asctime)s - %(funcName)s - %(levelname)s - %(message)s')
# handler.setFormatter(formatter)
# handler.setLevel(logging.INFO)
# logger.addHandler(handler)
import inspect
import traceback
import re
import collections


# Logging with the stack trace
def stack_log(msg):
    frame = inspect.currentframe()
    stack_trace = traceback.format_stack(frame)
    is_start_recording = False
    log_text = ""
    for i in stack_trace[:-1]:
        match = re.search('.*, line (.*), in (.*)', i)
        num_line = match.group(1)
        fun_name = match.group(2)
        if is_start_recording:
            log_text += fun_name + " " + num_line + " "
        if fun_name == "_test_edit":
            is_start_recording = True
    logger.info("[{}]".format(log_text) + msg)


# Third party imports
from qtpy.compat import from_qvariant, to_qvariant
from qtpy.QtCore import (QAbstractTableModel, QModelIndex, Qt, Signal, Slot,
                         QItemSelectionModel, QEvent)
from qtpy.QtGui import QColor, QCursor
from qtpy.QtWidgets import (QApplication, QCheckBox, QDialog, QGridLayout,
                            QHBoxLayout, QInputDialog, QLineEdit, QMenu,
                            QMessageBox, QPushButton, QTableView,
                            QScrollBar, QTableWidget, QFrame,
                            QItemDelegate, QHeaderView, QLabel)
from PyQt5.QtCore import pyqtSignal, QRect
from pandas import DataFrame, Index, Series, isna

try:
    from pandas._libs.tslib import OutOfBoundsDatetime
except ImportError:  # For pandas version < 0.20
    from pandas.tslib import OutOfBoundsDatetime
import numpy as np
import pandas as pd
# Local imports
from spyder.config.base import _
from spyder.config.fonts import DEFAULT_SMALL_DELTA
from spyder.config.gui import get_font, config_shortcut
from spyder.py3compat import (io, is_text_string, is_type_text_string, PY2,
                              to_text_string)
from spyder.utils import icon_manager as ima
from spyder.utils.qthelpers import (add_actions, create_action,
                                    keybinding, qapplication)
from spyder.plugins.variableexplorer.widgets.arrayeditor import get_idx_rect

# Supported Numbers and complex numbers
REAL_NUMBER_TYPES = (float, int, np.int64, np.int32, np.int16, np.float32, np.float64)
COMPLEX_NUMBER_TYPES = (complex, np.complex64, np.complex128)
# Used to convert bool intrance to false since bool('False') will return True
_bool_false = ['false', 'f', '0', '0.', '0.0', ' ']

# Default format for data frames with floats
DEFAULT_FORMAT = '%.6g'

# Background colours
BACKGROUND_NUMBER_MINHUE = 0.66  # hue for largest number
BACKGROUND_NUMBER_HUERANGE = 0.33  # (hue for smallest) minus (hue for largest)
BACKGROUND_NUMBER_SATURATION = 0.7
BACKGROUND_NUMBER_VALUE = 1.0
BACKGROUND_NUMBER_ALPHA = 0.6
BACKGROUND_NONNUMBER_COLOR = Qt.lightGray
BACKGROUND_INDEX_ALPHA = 0.8
BACKGROUND_STRING_ALPHA = 0.05
BACKGROUND_MISC_ALPHA = 0.3
""" API Function dictionary
sizeHint: The default implementation of sizeHint() returns an invalid size if there is no layout for this widget
          and returns the layout's preferred size otherwise.
updateGeometries: Updates the geometry of the child widgets of the view. called when sizeHint etc. is called
setViewportMargins: Sets the margins around the scrolling area to left, top, right and bottom. (for locked rows and columns)
sectionPosition: first visible item's top-left corner to the top-left corner of the item with logicalIndex
offset: header's left most visible pixel position
beginResetModel: When a model is reset it means that any previous data reported from the model is now invalid and has to be queried for again. 
                This also means that the current item and any selected items will become invalid.
"""

""" Layout workflow Original:
1. setup_and_check (initialization)
    1.1 setModel (setting the model, also used by sorting and filtering )
        _update_layout 
            setFixedHeight/setFixedWidth for level, index and header using sizeHint, rowViewportPosition etc
            _resizeVisibleColumnsToContents (resize the columns in view) 
                _resizeColumnToContents (set column width)
                    _sizeHintForColumn/sizeHintForIndex
                    setColumnWidth (on header)
                    
    1.2 resizeColumnsToContents 
        _resizeColumnsToContents (on index only)
        _update_layout
        table_level.resizeColumnsToContents (call API)
    
2. eventFilter (receive event of obj == self.dataTable and event.type() == QEvent.Resize )(called for any resize event)
    _resizeVisibleColumnsToContents 

"""
"""Layout workflow New: 
_resizeVisibleColumnsToContents -> _resizeAllColumnsToContents
"""

""" Layout related problem 
Problem 1: 
    Non-resized columns outside of initial view: 
    When scroll to and click on a column that was not in view initially,
    it resizes and causing the view to shift unwantedly.
Reason: 
    _resizeVisibleColumnsToContents resize only visible columns,
    sel_model.currentColumnChanged.connect(self._resizeCurrentColumnToContents) resizes on click
Solution:
    2018-03-11 connect scroll bar with signal sig_resize_columns and _resizeAllColumnsToContents
    2018-03-17 Set COLS_TO_LOAD to load all columns, _resizeVisibleColumnsToContents -> _resizeAllColumnsToContents  

Problem 2:
    Vertical scroll:
    Scrolling down the dataframe, the table view shows empty table for data beyond rows.loaded, until the scroll
    reaches the bottom then triggered load_more_data unexpectedly...
    This problem raised from multiple places that should not have affected it, changes from customheaderview, _update_layout, etc. 
Reason:
    Unknown  
Solution:
    by calling QApplication.processEvents() in setup_and_check 
    (Processes all pending events for the calling thread according to the specified flags until there are no more events to process.)

Problem 3:
    index column not resized properly
Reason:
    original _resizeColumnsToContents resizes based on contents from index and level. (we want to also resize it with customheader which does not belong to contents)
    Additionally, not all of the data was loaded to the view, so large index number was not used for resizing purpose 
Solution:
    In setup_and_check we call resizeIndexColumn which will call setColumnWidth based on both the header width and the width of the loaded index 
"""

############## Custom Header ########################
"""
This part creates a custom header build on top of the original QHeaderView
It includes:
1. For each column in data table, create an editor for filtering purpose 
2. For the index/level, a label which shows number of columns of the data table
Both of these would be right below column name/index name and should be aligned with them 
"""


# Source: https://stackoverflow.com/questions/44343738/how-to-inject-widgets-between-qheaderview-and-qtableview
class CustomHeaderView(QHeaderView):
    def __init__(self, parent):
        super().__init__(Qt.Horizontal, parent)
        self._is_initialized = False
        # since QLineEdit().sizeHint().height = 22
        self.CUSTOM_HEADER_HEIGHT = 22

    def sizeHint(self):
        size = super().sizeHint()
        size.setHeight(size.height() + self.CUSTOM_HEADER_HEIGHT)
        return size

    def updateGeometries(self):
        super().updateGeometries()
        self.setViewportMargins(0, 0, 0, self.CUSTOM_HEADER_HEIGHT)
        self.adjustPositions()


class CustomHeaderViewIndex(CustomHeaderView):
    def __init__(self, parent):
        super().__init__(parent)
        self._label = None
        self.sectionResized.connect(self.adjustPositions)

    def setQLabel(self, text):
        if not self._is_initialized:
            self._is_initialized = True
            self._label = QLabel(self.parent())
            self._label.setStyleSheet('color: blue')
            self._label.setAlignment(Qt.AlignCenter | Qt.AlignVCenter)
            self.adjustPositions()
        self._label.setText(text)

    def adjustPositions(self):
        if self._is_initialized:
            x = self.sectionPosition(0) - self.offset()
            # check the original header height
            y = super(CustomHeaderView, self).sizeHint().height()
            self._label.setGeometry(QRect(x, y, self.sectionSize(0), self.CUSTOM_HEADER_HEIGHT))

    def sizeHint(self):
        size = super().sizeHint()
        if self._is_initialized:
            label_width = self._label.sizeHint().width()
            size.setWidth(label_width)
        return size


class CustomHeaderViewEditor(CustomHeaderView):
    filterActivated = Signal()

    def __init__(self, parent):
        super().__init__(parent)
        self._editors = []
        self.sectionResized.connect(self.adjustPositions)
        parent.horizontalScrollBar().valueChanged.connect(self.adjustPositions)

    # connected with sectionResized and horizontalScrollBar
    def adjustPositions(self):
        for index, editor in enumerate(self._editors):
            # offset: header's left most visible pixel position
            x = self.sectionPosition(index) - self.offset()
            # check the original header height
            y = super(CustomHeaderView, self).sizeHint().height()
            # move to x,y position
            # coordinates is left and top most of the structure
            editor.move(x, y)
            editor.resize(self.sectionSize(index), self.CUSTOM_HEADER_HEIGHT)

    def setFilterBoxes(self, count):
        # create only once
        if not self._is_initialized:
            for index in range(count):
                editor = QLineEdit(self.parent())
                editor.setPlaceholderText(str(index))
                # TODO: shift return to represent union
                editor.returnPressed.connect(self.filterActivated.emit)
                self._editors.append(editor)
            self._is_initialized = True
            self.adjustPositions()

    def getText(self):
        return [self._editors[index].text() for index in range(0, len(self._editors))]

    def _clearFilterText(self):
        for index in range(0, len(self._editors)):
            self._editors[index].setText('')


############## Custom Header ########################
def bool_false_check(value):
    """
    Used to convert bool entrance to false.

    Needed since any string in bool('') will return True.
    """
    if value.lower() in _bool_false:
        value = ''
    return value


def global_max(col_vals, index):
    """Returns the global maximum and minimum."""
    col_vals_without_None = [x for x in col_vals if x is not None]
    max_col, min_col = zip(*col_vals_without_None)
    return max(max_col), min(min_col)


class DataFrameModel(QAbstractTableModel):
    """ DataFrame Table Model.

    Partly based in ExtDataModel and ExtFrameModel classes
    of the gtabview project.

    For more information please see:
    https://github.com/wavexx/gtabview/blob/master/gtabview/models.py
    """

    def __init__(self, dataFrame, format=DEFAULT_FORMAT, parent=None):
        # model loading values
        # Limit at which dataframe is considered large and it is loaded on demand
        self.LARGE_SIZE = 5e5
        # self.LARGE_SIZE = 5e6
        self.LARGE_NROWS = 1e5
        self.LARGE_COLS = 60
        self.ROWS_TO_LOAD = 500
        self.COLS_TO_LOAD = dataFrame.shape[1]
        self.UNIQUE_ITEM_THRESHOLD = 15

        QAbstractTableModel.__init__(self)
        self.dialog = parent
        self.df = dataFrame
        self.original_df = dataFrame.copy()
        self.idx_tracker = None
        self._INDEX_TRACKER_NAME = '__INDEX_TRACKER'
        self.df_index = dataFrame.index.tolist()
        self.df_header = dataFrame.columns.tolist()
        self._format = format
        self.complex_intran = None
        self.display_error_idxs = []

        self.total_rows = self.df.shape[0]
        self.total_cols = self.df.shape[1]
        size = self.total_rows * self.total_cols
        self.filtered_text = ''

        self.unique_items_col = [None] * self.df.shape[1]
        self.max_min_col = None
        if size < self.LARGE_SIZE:
            self.update_df_features()
            self.colum_avg_enabled = True
            self.bgcolor_enabled = True
            self.colum_avg(1)
        else:
            self.colum_avg_enabled = False
            self.bgcolor_enabled = False
            self.colum_avg(0)

        # Use paging when the total size, number of rows or number of
        # columns is too large
        if size > self.LARGE_SIZE:
            self.rows_loaded = self.ROWS_TO_LOAD
            self.cols_loaded = self.COLS_TO_LOAD
        else:
            if self.total_rows > self.LARGE_NROWS:
                self.rows_loaded = self.ROWS_TO_LOAD
            else:
                self.rows_loaded = self.total_rows
            if self.total_cols > self.LARGE_COLS:
                self.cols_loaded = self.COLS_TO_LOAD
            else:
                self.cols_loaded = self.total_cols

    def _axis(self, axis):
        """
        Return the corresponding labels taking into account the axis.

        The axis could be horizontal (0) or vertical (1).
        """
        return self.df.columns if axis == 0 else self.df.index

    def _axis_levels(self, axis):
        """
        Return the number of levels in the labels taking into account the axis.

        Get the number of levels for the columns (0) or rows (1).
        """
        ax = self._axis(axis)
        return 1 if not hasattr(ax, 'levels') else len(ax.levels)

    @property
    def shape(self):
        """Return the shape of the dataframe."""
        return self.df.shape

    @property
    def header_shape(self):
        """Return the levels for the columns and rows of the dataframe."""
        return (self._axis_levels(0), self._axis_levels(1))

    @property
    def chunk_size(self):
        """Return the max value of the dimensions of the dataframe."""
        return max(*self.shape())

    def header(self, axis, x, level=0):
        """
        Return the values of the labels for the header of columns or rows.

        The value corresponds to the header of column or row x in the
        given level.
        """
        ax = self._axis(axis)
        return ax.values[x] if not hasattr(ax, 'levels') \
            else ax.values[x][level]

    def name(self, axis, level):
        """Return the labels of the levels if any."""
        ax = self._axis(axis)
        if hasattr(ax, 'levels'):
            return ax.names[level]
        if ax.name:
            return ax.name

    def unique_col_update(self, col_idx=None):
        """return list of sorted unique values of each column, None for unhashable values"""
        if self.df.shape[0] == 0:  # If no rows to compute max/min then return
            return
        for idx, col in enumerate(self.df):
            if col_idx is None or (col_idx is not None and col_idx == idx):
                unique_items = None
                try:
                    # TODO maybe check unique size first?
                    unique_items = self.df[col].unique().tolist()
                    unique_items = sorted(unique_items)
                except TypeError:
                    # unhashable values OR mixed data type could not be sorted
                    pass
                self.unique_items_col[idx] = unique_items

    def max_min_col_update(self):
        """
        Determines the maximum and minimum number in each column.

        The result is a list whose k-th entry is [vmax, vmin], where vmax and
        vmin denote the maximum and minimum of the k-th column (ignoring NaN). 
        This list is stored in self.max_min_col.

        If the k-th column has a non-numerical dtype, then the k-th entry
        is set to None. If the dtype is complex, then compute the maximum and
        minimum of the absolute values. If vmax equals vmin, then vmin is 
        decreased by one.
        """
        if self.df.shape[0] == 0:  # If no rows to compute max/min then return
            return
        self.max_min_col = [None] * self.df.shape[1]
        for idx, (_, col) in enumerate(self.df.iteritems()):
            vmax = vmin = None  # set default values
            unique_items = self.unique_items_col[idx]  # check unique items
            if unique_items is not None:
                # set min_max to None (no color) for single value column
                if len(unique_items) == 1:
                    continue
            if col.dtype in REAL_NUMBER_TYPES:
                vmax = col.max(skipna=True)
                vmin = col.min(skipna=True)
            elif col.dtype in COMPLEX_NUMBER_TYPES:
                vmax = col.abs().max(skipna=True)
                vmin = col.abs().min(skipna=True)
            else:
                if unique_items is not None:
                    if len(unique_items) < self.UNIQUE_ITEM_THRESHOLD:
                        vmax = len(unique_items) - 1
                        vmin = 0
            if vmax is not None and vmin is not None:
                if vmax != vmin:
                    max_min = [vmax, vmin]
                else:
                    # reach here iff only column only contains one value and nan
                    max_min = [vmax, vmin - 1]
                self.max_min_col[idx] = max_min

    def get_format(self):
        """Return current format"""
        # Avoid accessing the private attribute _format from outside
        return self._format

    def set_format(self, format):
        """Change display format"""
        self._format = format
        self.reset()

    def bgcolor(self, state):
        """Toggle backgroundcolor"""
        self.bgcolor_enabled = state > 0
        self.reset()

    def colum_avg(self, state):
        """Toggle backgroundcolor"""
        self.colum_avg_enabled = state > 0
        if self.colum_avg_enabled:
            self.return_max = lambda col_vals, index: col_vals[index]
        else:
            self.return_max = global_max
        self.reset()

    def set_filter(self, filter_list, df_name):
        """filter self.df by the filter_list given
        just like sort, modify self.df and then reset"""
        filter_per_column = []
        for idx, filter_str in enumerate(filter_list):
            result = self._get_query_list(idx, filter_str)
            if result is not None:
                filter_per_column.append(result)
        query_text = '&'.join(filter_per_column)
        if query_text == '':
            # empty filter
            self.df = self.original_df.copy()
            self.filtered_text = ''
            self.idx_tracker = None
        else:
            self.set_index_tracker(self.original_df, is_reset=True)
            core_exec_text = 'self.original_df[{}]'.format(query_text)
            exec_text = 'self.df = ' + core_exec_text + '.copy()'
            try:
                exec(exec_text)
            except:
                QMessageBox.critical(self.dialog, _("Error"), traceback.format_exc())
                self.remove_index_tracker(self.original_df, is_update=False)
                return
            self.filtered_text = core_exec_text.replace('self.original_df', df_name)
            self.remove_index_tracker(self.original_df, is_update=False)
            self.remove_index_tracker(self.df, is_update=True)
        # reset total rows
        self.total_rows = self.df.shape[0]
        self.update_df_features()
        self.reset()

    def _get_query_list(self, idx, filter_str):
        """wrap filter_str with df name and column name with special characters handling

        (Assume "|" and "&" not used other than separator)
        Multiple logical statements within a editing cell (column) could be separated by "|" OR "&" which represent union OR intersection
        They could not be present at the same time
        Each statement should START with either the logical operators: > < !=  ==
        OR the following definitions:
        ^ : replace by .str.startswith
        ISNAN: replace by .np.isnan()

        If none of above is detected, will use == by default

        Example of filter_str: 1|2  >5&<10  ^"E"
        """

        def _get_handled_logic(logic_str, column_name):
            logic_str = logic_str.strip()
            if logic_str.startswith((">", "<", "!=", "==")):
                pass
            elif logic_str.startswith("^"):
                logic_str = ".str.startswith({})".format(logic_str[1:])
            elif logic_str == "ISNAN":
                return '(pd.isnull(self.original_df["{}"]))'.format(column_name)
            elif logic_str == "!ISNAN":
                return '(~pd.isnull(self.original_df["{}"]))'.format(column_name)
            else:
                # default
                logic_str = "==" + logic_str
            return '(self.original_df["{}"]{})'.format(column_name, logic_str)

        #####
        if filter_str == '':
            return
        column_name = str(self.original_df.columns[idx])
        assert (not ("|" in filter_str and "&" in filter_str))
        for i in ["|", "&"]:
            if i in filter_str:
                multiple_logic = [_get_handled_logic(filter_str, column_name) for filter_str in filter_str.split(i)]
                return i.join(multiple_logic)
        else:
            return _get_handled_logic(filter_str, column_name)

    # this is the Qmodelindex http://doc.qt.io/qt-5/qmodelindex.html
    def get_bgcolor(self, index):
        """Background color depending on value."""
        column = index.column()
        if not self.bgcolor_enabled:
            return
        value = self.get_value(index.row(), column)
        if self.max_min_col[column] is None or isna(value):
            color = QColor(BACKGROUND_NONNUMBER_COLOR)
            if is_text_string(value):
                # transparency
                color.setAlphaF(BACKGROUND_STRING_ALPHA)
            else:
                color.setAlphaF(BACKGROUND_MISC_ALPHA)
        else:
            if isinstance(value, COMPLEX_NUMBER_TYPES):
                color_func = abs
            elif isinstance(value, REAL_NUMBER_TYPES):
                color_func = float
            # other objects, just like int
            else:
                color_func = float
                if self.unique_items_col[column] is not None:
                    try:
                        # transform the value to index of unique items
                        value = self.unique_items_col[column].index(value)
                    except ValueError:
                        #  datetime are transformed when called by series.unique(), cannot match with unique_items_col
                        return
                else:
                    return
            # self.return_max returns global max or column max
            vmax, vmin = self.return_max(self.max_min_col, column)
            # handle infinity values
            if value == np.inf:
                hue_ratio = 0
            elif value == -np.inf:
                hue_ratio = 1
            else:
                if np.isinf(vmax) or np.isinf(vmin):
                    # if max or min is inf, default color for other values
                    return
                else:
                    hue_ratio = (vmax - color_func(value)) / (vmax - vmin)
            hue = BACKGROUND_NUMBER_MINHUE + BACKGROUND_NUMBER_HUERANGE * hue_ratio
            hue = float(abs(hue))
            if hue > 1:
                hue = 1
            color = QColor.fromHsvF(hue, BACKGROUND_NUMBER_SATURATION,
                                    BACKGROUND_NUMBER_VALUE,
                                    BACKGROUND_NUMBER_ALPHA)
        return color

    def get_value(self, row, column):
        """Return the value of the DataFrame."""
        # To increase the performance iat is used but that requires error
        # handling, so fallback uses iloc
        try:
            value = self.df.iat[row, column]
        except OutOfBoundsDatetime:
            value = self.df.iloc[:, column].astype(str).iat[row]
        except:
            value = self.df.iloc[row, column]
        return value

    def update_df_index(self):
        """"Update the DataFrame index"""
        self.df_index = self.df.index.tolist()

    def data(self, index, role=Qt.DisplayRole):
        """Cell content"""
        if not index.isValid():
            return to_qvariant()
        if role == Qt.DisplayRole or role == Qt.EditRole:
            column = index.column()
            row = index.row()
            value = self.get_value(row, column)
            if isinstance(value, float):
                try:
                    return to_qvariant(self._format % value)
                except (ValueError, TypeError):
                    # may happen if format = '%d' and value = NaN;
                    # see issue 4139
                    return to_qvariant(DEFAULT_FORMAT % value)
            elif is_type_text_string(value):
                # Don't perform any conversion on strings
                # because it leads to differences between
                # the data present in the dataframe and
                # what is shown by Spyder
                return value
            else:
                try:
                    return to_qvariant(to_text_string(value))
                except Exception:
                    self.display_error_idxs.append(index)
                    return u'Display Error!'
        elif role == Qt.BackgroundColorRole:
            return to_qvariant(self.get_bgcolor(index))
        elif role == Qt.FontRole:
            return to_qvariant(get_font(font_size_delta=DEFAULT_SMALL_DELTA))
        elif role == Qt.ToolTipRole:
            if index in self.display_error_idxs:
                return _("It is not possible to display this value because\n"
                         "an error ocurred while trying to do it")
        return to_qvariant()

    def sort(self, column, order=Qt.AscendingOrder):
        """Overriding sort method"""
        if self.complex_intran is not None:
            if self.complex_intran.any(axis=0).iloc[column]:
                QMessageBox.critical(self.dialog, "Error",
                                     "TypeError error: no ordering "
                                     "relation is defined for complex numbers")
                return False
        self.set_index_tracker(self.df, is_reset=False)
        try:
            ascending = order == Qt.AscendingOrder
            if column >= 0:
                try:
                    self.df.sort_values(by=self.df.columns[column],
                                        ascending=ascending, inplace=True,
                                        kind='mergesort')
                except AttributeError:
                    # for pandas version < 0.17
                    self.df.sort(columns=self.df.columns[column],
                                 ascending=ascending, inplace=True,
                                 kind='mergesort')
                except ValueError as e:
                    # Not possible to sort on duplicate columns #5225
                    QMessageBox.critical(self.dialog, "Error",
                                         "ValueError: %s" % to_text_string(e))
                except SystemError as e:
                    # Not possible to sort on category dtypes #5361
                    QMessageBox.critical(self.dialog, "Error",
                                         "SystemError: %s" % to_text_string(e))
                self.update_df_index()
            else:
                # To sort by index
                self.df.sort_index(inplace=True, ascending=ascending)
                self.update_df_index()
        except TypeError as e:
            QMessageBox.critical(self.dialog, "Error",
                                 "TypeError error: %s" % str(e))
            self.remove_index_tracker(self.df, is_update=False)
            return False
        self.remove_index_tracker(self.df, is_update=True)
        self.reset()
        return True

    def find_next_value_same_col(self, cur_row, cur_col, is_direction_up):
        cur_val = self.get_value(cur_row, cur_col)
        selected_col = self.df.iloc[:, cur_col]
        if not isinstance(cur_val, str) and isinstance(cur_val, collections.Iterable):
            # do not compare if its a list/dict/np.array
            index_candidates = np.array([])
        else:
            # check nan
            if pd.isna(cur_val):
                index_candidates = np.where(~pd.isnull(selected_col))[0]
            else:
                index_candidates = np.where(selected_col != cur_val)[0]
        if is_direction_up:
            try:
                return index_candidates[index_candidates < cur_row][-1]
            except IndexError:
                return 0
        else:
            try:
                return index_candidates[index_candidates > cur_row][0]
            except IndexError:
                return self.df.shape[0] - 1

    def flags(self, index):
        """Set flags"""
        return Qt.ItemFlags(QAbstractTableModel.flags(self, index) |
                            Qt.ItemIsEditable)

    def setData(self, index, value, role=Qt.EditRole, change_type=None):
        """Cell content change"""
        column = index.column()
        row = index.row()

        if index in self.display_error_idxs:
            return False
        if change_type is not None:
            try:
                value = self.data(index, role=Qt.DisplayRole)
                val = from_qvariant(value, str)
                if change_type is bool:
                    val = bool_false_check(val)
                self.df.iloc[row, column] = change_type(val)
            except ValueError:
                self.df.iloc[row, column] = change_type('0')
        else:
            val = from_qvariant(value, str)
            current_value = self.get_value(row, column)
            if isinstance(current_value, (bool, np.bool_)):
                val = bool_false_check(val)
            supported_types = (bool, np.bool_) + REAL_NUMBER_TYPES
            if (isinstance(current_value, supported_types) or
                    is_text_string(current_value)):
                try:
                    val_to_set = current_value.__class__(val)
                    # skip the updating when no changes, also check value is nan
                    if val_to_set == current_value or ((val_to_set != val_to_set) and (current_value != current_value)):
                        return True
                    self.df.iloc[row, column] = val_to_set
                except (ValueError, OverflowError) as e:
                    QMessageBox.critical(self.dialog, "Error",
                                         str(type(e).__name__) + ": " + str(e))
                    return False
            else:
                QMessageBox.critical(self.dialog, "Error",
                                     "Editing dtype {0!s} not yet supported."
                                     .format(type(current_value).__name__))
                return False
        self.max_min_col_update()
        self.dataChanged.emit(index, index)
        return True

    def get_data(self):
        """Return data"""
        return self.df

    def rowCount(self, index=QModelIndex()):
        """DataFrame row number"""
        # Avoid a "Qt exception in virtual methods" generated in our
        # tests on Windows/Python 3.7
        # See PR 8910
        try:
            if self.total_rows <= self.rows_loaded:
                return self.total_rows
            else:
                return self.rows_loaded
        except AttributeError:
            return 0

    def fetch_more(self, rows=False, columns=False):
        # This is called when scroll reaches the maximum, load_more_data is called,
        # which fetches more data for the model and then fetch for the index/header
        """Get more columns and/or rows."""
        if rows and self.total_rows > self.rows_loaded:
            reminder = self.total_rows - self.rows_loaded
            items_to_fetch = min(reminder, self.ROWS_TO_LOAD)
            self.beginInsertRows(QModelIndex(), self.rows_loaded,
                                 self.rows_loaded + items_to_fetch - 1)
            self.rows_loaded += items_to_fetch
            self.endInsertRows()
        if columns and self.total_cols > self.cols_loaded:
            reminder = self.total_cols - self.cols_loaded
            items_to_fetch = min(reminder, self.COLS_TO_LOAD)
            self.beginInsertColumns(QModelIndex(), self.cols_loaded,
                                    self.cols_loaded + items_to_fetch - 1)
            self.cols_loaded += items_to_fetch
            self.endInsertColumns()

    def columnCount(self, index=QModelIndex()):
        """DataFrame column number"""
        # Avoid a "Qt exception in virtual methods" generated in our
        # tests on Windows/Python 3.7
        # See PR 8910
        try:
            # This is done to implement series
            if len(self.df.shape) == 1:
                return 2
            elif self.total_cols <= self.cols_loaded:
                return self.total_cols
            else:
                return self.cols_loaded
        except AttributeError:
            return 0

    def reset(self):
        self.beginResetModel()
        self.endResetModel()

    def update_df_features(self, col_idx=None):
        self.unique_col_update(col_idx)
        self.max_min_col_update()

    def set_rows_to_load(self, rows_to_load):
        self.ROWS_TO_LOAD = rows_to_load

    def set_cols_to_load(self, cols_to_load):
        self.COLS_TO_LOAD = cols_to_load

    def set_index_tracker(self, df, is_reset):
        if self.idx_tracker is None or is_reset:
            df[self._INDEX_TRACKER_NAME] = range(len(df))
        else:
            # when index is duplicated,  set directly would not work
            df[self._INDEX_TRACKER_NAME] = self.idx_tracker.values

    def remove_index_tracker(self, df, is_update):
        if is_update:
            self.idx_tracker = df[self._INDEX_TRACKER_NAME]
        df.drop(self._INDEX_TRACKER_NAME, axis=1, inplace=True)


class DataFrameView(QTableView):
    """
    Data Frame view class.

    Signals
    -------
    sig_option_changed(): Raised after a sort by column.
    sig_sort_by_column(): Raised after more columns are fetched.
    sig_fetch_more_rows(): Raised after more rows are fetched.
    """
    sig_sort_by_column = Signal()
    sig_fetch_more_columns = Signal()
    sig_fetch_more_rows = Signal()
    sig_plot = Signal()
    sig_subplot = Signal()
    sig_reset_and_scroll_to = Signal()
    sig_reset_sort_indicator = Signal()

    def __init__(self, parent, model, header, hscroll, vscroll):
        """Constructor."""
        QTableView.__init__(self, parent)
        self.setModel(model)
        self.setHorizontalScrollBar(hscroll)
        self.setVerticalScrollBar(vscroll)
        # set to scrollperpixel, (The view will scroll the contents one pixel at a time)
        self.setHorizontalScrollMode(1)
        self.setVerticalScrollMode(1)

        self.sort_old = [None]
        #  header_class is headerview. horizontalHeader()
        self.header_class = header
        self.header_class.sectionClicked.connect(self.sortByColumn)
        self.menu = self.setup_menu()
        config_shortcut(self.copy, context='variable_explorer', name='copy',
                        parent=self)
        self.horizontalScrollBar().valueChanged.connect(
            lambda val: self.load_more_data(val, columns=True))
        self.verticalScrollBar().valueChanged.connect(
            lambda val: self.load_more_data(val, rows=True))

    def load_more_data(self, value, rows=False, columns=False):
        """Load more rows and columns to display."""
        try:
            if rows and value == self.verticalScrollBar().maximum():
                self.model().fetch_more(rows=rows)
                self.sig_fetch_more_rows.emit()
            """
            if columns and value == self.horizontalScrollBar().maximum():
                self.model().fetch_more(columns=columns)
                self.sig_fetch_more_columns.emit()
            """
        except NameError:
            # Needed to handle a NameError while fetching data when closing
            # See issue 7880
            pass

    def keyPressEvent(self, event):
        modifiers = QApplication.keyboardModifiers()
        # if pressed control
        if modifiers == Qt.ControlModifier:
            current_row = self.selectionModel().currentIndex().row()
            current_col = self.selectionModel().currentIndex().column()
            if event.key() == Qt.Key_Up:
                self.scroll_to_and_select(0, current_col)
            if event.key() == Qt.Key_Left:
                self.scroll_to_and_select(current_row, 0)
            if event.key() == Qt.Key_Down:
                self.scroll_to_and_select(self.model().total_rows - 1, current_col)
            if event.key() == Qt.Key_Right:
                self.scroll_to_and_select(current_row, self.model().total_cols - 1)
            if event.key() in [Qt.Key_Up, Qt.Key_Left, Qt.Key_Down, Qt.Key_Right]:
                # override original implementation
                return
        if modifiers == Qt.AltModifier:
            current_row = self.selectionModel().currentIndex().row()
            current_col = self.selectionModel().currentIndex().column()
            if event.key() == Qt.Key_Up:
                self.go_to_next_value_same_col(current_row, current_col, True)
            if event.key() == Qt.Key_Down:
                # self.scroll_to_and_select(self.model().total_rows - 1, current_col)
                self.go_to_next_value_same_col(current_row, current_col, False)
            if event.key() in [Qt.Key_Up, Qt.Key_Down]:
                return
        super(DataFrameView, self).keyPressEvent(event)

    def scroll_to_and_select(self, row, col):
        if row > self.model().rows_loaded:
            self.model().set_rows_to_load(row)
            self.model().fetch_more(rows=True)
            self.sig_fetch_more_rows.emit()
        if col > self.model().cols_loaded:
            self.model().set_cols_to_load(col)
            self.model().fetch_more(columns=True)
            self.sig_fetch_more_columns.emit()
        # let the view update
        QApplication.processEvents()
        self.scrollTo(self.model().index(row, col))
        self.selectionModel().setCurrentIndex(self.model().index(row, col), QItemSelectionModel.ClearAndSelect)

    def go_to_next_value_same_col(self, cur_row, cur_col, is_direction_up):
        """scroll to the next item in this column which value is not the same"""
        row = self.model().find_next_value_same_col(cur_row, cur_col, is_direction_up)
        self.scroll_to_and_select(row, cur_col)

    def sortByColumn(self, index):
        """Implement a column sort."""
        if self.sort_old == [None]:
            self.header_class.setSortIndicatorShown(True)
            # the arrow in header
        sort_order = self.header_class.sortIndicatorOrder()
        self.sig_sort_by_column.emit()
        if not self.model().sort(index, sort_order):
            if len(self.sort_old) != 2:
                self.header_class.setSortIndicatorShown(False)
            else:
                self.header_class.setSortIndicator(self.sort_old[0],
                                                   self.sort_old[1])
            return
        self.sort_old = [index, self.header_class.sortIndicatorOrder()]
        self.sig_reset_sort_indicator.emit()

    def _reset_sort_indicator(self):
        self.header_class.setSortIndicatorShown(False)
        self.sort_old = [None]

    def contextMenuEvent(self, event):
        """Reimplement Qt method."""
        self.menu.popup(event.globalPos())
        event.accept()

    def setup_menu(self):
        """Setup context menu."""
        copy_action = create_action(self, _('Copy'),
                                    shortcut=keybinding('Copy'),
                                    icon=ima.icon('editcopy'),
                                    triggered=self.copy,
                                    context=Qt.WidgetShortcut)
        plot_action = create_action(self, _('Plot selected columns'),
                                    icon=ima.icon('plot'),
                                    triggered=self.sig_plot,
                                    context=Qt.WidgetShortcut)
        subplot_action = create_action(self, _('Plot selected columns with sublplots'),
                                       icon=ima.icon('plot'),
                                       triggered=self.sig_subplot,
                                       context=Qt.WidgetShortcut)
        reset_action = create_action(self, _('Reset and scroll to'),
                                     icon=ima.icon('restore'),
                                     triggered=self.sig_reset_and_scroll_to,
                                     context=Qt.WidgetShortcut)
        functions = ((_("To bool"), bool), (_("To complex"), complex),
                     (_("To int"), int), (_("To float"), float),
                     (_("To str"), to_text_string))
        types_in_menu = [copy_action]
        types_in_menu += [plot_action, subplot_action, reset_action]
        for name, func in functions:
            slot = lambda func=func: self.change_type(func)
            types_in_menu += [create_action(self, name,
                                            triggered=slot,
                                            context=Qt.WidgetShortcut)]
        menu = QMenu(self)
        add_actions(menu, types_in_menu)
        return menu

    def change_type(self, func):
        """A function that changes types of cells."""
        model = self.model()
        index_list = self.selectedIndexes()
        [model.setData(i, '', change_type=func) for i in index_list]

    @Slot()
    def copy(self):
        """Copy text to clipboard"""
        if not self.selectedIndexes():
            return
        (row_min, row_max,
         col_min, col_max) = get_idx_rect(self.selectedIndexes())
        index = header = False
        df = self.model().df
        obj = df.iloc[slice(row_min, row_max + 1),
                      slice(col_min, col_max + 1)]
        output = io.StringIO()
        obj.to_csv(output, sep='\t', index=index, header=header)
        if not PY2:
            contents = output.getvalue()
        else:
            contents = output.getvalue().decode('utf-8')
        output.close()
        clipboard = QApplication.clipboard()
        clipboard.setText(contents)


class DataFrameHeaderModel(QAbstractTableModel):
    """
    This class is the model for the header or index of the DataFrameEditor.

    Taken from gtabview project (Header4ExtModel).
    For more information please see:
    https://github.com/wavexx/gtabview/blob/master/gtabview/viewer.py
    """

    COLUMN_INDEX = -1  # Makes reference to the index of the table.

    def __init__(self, model, axis, palette):
        """
        Header constructor.

        The 'model' is the QAbstractTableModel of the dataframe, the 'axis' is
        to acknowledge if is for the header (horizontal - 0) or for the
        index (vertical - 1) and the palette is the set of colors to use.
        """
        super(DataFrameHeaderModel, self).__init__()
        self.model = model
        self.axis = axis
        self._palette = palette
        # all sizes related variables should just follow DataFrameModel
        if self.axis == 0:
            self.total_cols = self.model.total_cols
            self._shape = (self.model.header_shape[0], self.model.shape[1])
            self.cols_loaded = self.model.cols_loaded
        else:
            self.total_rows = self.model.total_rows
            self._shape = (self.model.shape[0], self.model.header_shape[1])
            self.rows_loaded = self.model.rows_loaded

    def rowCount(self, index=None):
        """Get number of rows in the header."""
        if self.axis == 0:
            return max(1, self._shape[0])
        else:
            if self.total_rows <= self.rows_loaded:
                return self.total_rows
            else:
                return self.rows_loaded

    def columnCount(self, index=QModelIndex()):
        """DataFrame column number"""
        if self.axis == 0:
            if self.total_cols <= self.cols_loaded:
                return self.total_cols
            else:
                return self.cols_loaded
        else:
            return max(1, self._shape[1])

    def fetch_more(self, rows=False, columns=False):
        """Get more columns or rows (based on axis)."""
        if self.axis == 1 and self.total_rows > self.rows_loaded:
            reminder = self.total_rows - self.rows_loaded
            items_to_fetch = min(reminder, self.model.ROWS_TO_LOAD)
            self.beginInsertRows(QModelIndex(), self.rows_loaded,
                                 self.rows_loaded + items_to_fetch - 1)
            self.rows_loaded += items_to_fetch
            self.endInsertRows()
        if self.axis == 0 and self.total_cols > self.cols_loaded:
            reminder = self.total_cols - self.cols_loaded
            items_to_fetch = min(reminder, self.model.COLS_TO_LOAD)
            self.beginInsertColumns(QModelIndex(), self.cols_loaded,
                                    self.cols_loaded + items_to_fetch - 1)
            self.cols_loaded += items_to_fetch
            self.endInsertColumns()

    def sort(self, column, order=Qt.AscendingOrder):
        """Overriding sort method."""
        ascending = order == Qt.AscendingOrder
        self.model.sort(self.COLUMN_INDEX, order=ascending)
        return True

    def headerData(self, section, orientation, role):
        """Get the information to put in the header."""
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return Qt.AlignCenter | Qt.AlignBottom
            else:
                return Qt.AlignRight | Qt.AlignVCenter
        if role != Qt.DisplayRole and role != Qt.ToolTipRole:
            return None
        if self.axis == 1 and self._shape[1] <= 1:
            return None
        orient_axis = 0 if orientation == Qt.Horizontal else 1
        if self.model.header_shape[orient_axis] > 1:
            header = section
        else:
            header = self.model.header(self.axis, section)

            # Don't perform any conversion on strings
            # because it leads to differences between
            # the data present in the dataframe and
            # what is shown by Spyder
            if not is_type_text_string(header):
                header = to_text_string(header)

        return header

    def data(self, index, role):
        """
        Get the data for the header.

        This is used when a header has levels.
        """
        if not index.isValid() or \
                index.row() >= self._shape[0] or \
                index.column() >= self._shape[1]:
            return None
        row, col = ((index.row(), index.column()) if self.axis == 0
                    else (index.column(), index.row()))
        if role != Qt.DisplayRole:
            return None
        if self.axis == 0 and self._shape[0] <= 1:
            return None

        header = self.model.header(self.axis, col, row)

        # Don't perform any conversion on strings
        # because it leads to differences between
        # the data present in the dataframe and
        # what is shown by Spyder
        if not is_type_text_string(header):
            header = to_text_string(header)

        return header


class DataFrameLevelModel(QAbstractTableModel):
    """
    Data Frame level class.

    This class is used to represent index levels in the DataFrameEditor. When
    using MultiIndex, this model creates labels for the index/header as Index i
    for each section in the index/header

    Based on the gtabview project (Level4ExtModel).
    For more information please see:
    https://github.com/wavexx/gtabview/blob/master/gtabview/viewer.py
    """

    def __init__(self, model, palette, font):
        super(DataFrameLevelModel, self).__init__()
        self.model = model
        self._background = palette.dark().color()
        if self._background.lightness() > 127:
            self._foreground = palette.text()
        else:
            self._foreground = palette.highlightedText()
        self._palette = palette
        font.setBold(True)
        self._font = font

    def rowCount(self, index=None):
        """Get number of rows (number of levels for the header)."""
        return max(1, self.model.header_shape[0])

    def columnCount(self, index=None):
        """Get the number of columns (number of levels for the index)."""
        return max(1, self.model.header_shape[1])

    def headerData(self, section, orientation, role):
        """
        Get the text to put in the header of the levels of the indexes.

        By default it returns 'Index i', where i is the section in the index
        """
        if role == Qt.TextAlignmentRole:
            if orientation == Qt.Horizontal:
                return Qt.AlignCenter | Qt.AlignBottom
            else:
                return Qt.AlignRight | Qt.AlignVCenter
        if role != Qt.DisplayRole and role != Qt.ToolTipRole:
            return None
        if self.model.header_shape[0] <= 1 and orientation == Qt.Horizontal:
            if self.model.name(1, section):
                return self.model.name(1, section)
            return _('Index')
        elif self.model.header_shape[0] <= 1:
            return None
        elif self.model.header_shape[1] <= 1 and orientation == Qt.Vertical:
            return None
        return _('Index') + ' ' + to_text_string(section)

    def data(self, index, role):
        """Get the information of the levels."""
        if not index.isValid():
            return None
        if role == Qt.FontRole:
            return self._font
        label = ''
        if index.column() == self.model.header_shape[1] - 1:
            label = str(self.model.name(0, index.row()))
        elif index.row() == self.model.header_shape[0] - 1:
            label = str(self.model.name(1, index.column()))
        if role == Qt.DisplayRole and label:
            return label
        elif role == Qt.ForegroundRole:
            return self._foreground
        elif role == Qt.BackgroundRole:
            return self._background
        elif role == Qt.BackgroundRole:
            return self._palette.window()
        return None


class DataFrameEditor(QDialog):
    """
    Dialog for displaying and editing DataFrame and related objects.

    Based on the gtabview project (ExtTableView).
    For more information please see:
    https://github.com/wavexx/gtabview/blob/master/gtabview/viewer.py

    Signals
    -------
    sig_option_changed(str, object): Raised if an option is changed.
       Arguments are name of option and its new value.
    """
    sig_option_changed = Signal(str, object)

    def __init__(self, parent=None):
        QDialog.__init__(self, parent)
        # Destroying the C++ object right after closing the dialog box,
        # otherwise it may be garbage-collected in another QThread
        # (e.g. the editor's analysis thread in Spyder), thus leading to
        # a segmentation fault on UNIX or an application crash on Windows
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.is_series = False
        self.layout = None

    def setup_and_check(self, data, title=''):
        """
        Setup DataFrameEditor:
        return False if data is not supported, True otherwise.
        Supported types for data are DataFrame, Series and Index.
        """
        # TODO: block multi index
        # TODOL block empty index
        self._selection_rec = False
        self._model = None

        self.layout = QGridLayout()
        self.layout.setSpacing(0)
        self.layout.setContentsMargins(0, 0, 0, 0)
        self.setLayout(self.layout)
        self.setWindowIcon(ima.icon('arredit'))
        # title = name of the dataframe in python
        self.df_name = title if title else "df"
        if title:
            title = to_text_string(title) + " - %s" % data.__class__.__name__
        else:
            title = _("%s editor") % data.__class__.__name__
        if isinstance(data, Series):
            self.is_series = True
            data = data.to_frame()
        elif isinstance(data, Index):
            data = DataFrame(data)

        self.setWindowTitle(title)
        self.resize(600, 500)

        self.hscroll = QScrollBar(Qt.Horizontal)
        self.vscroll = QScrollBar(Qt.Vertical)
        grey_scroll_style_sheet = """ 
                        QScrollBar {
                            background:grey;
                        }
                        QScrollBar::add-page{
                            background:white;
                        }
                        QScrollBar::sub-page{
                            background:white;
                        }
                        """
        self.hscroll.setStyleSheet(grey_scroll_style_sheet)
        self.vscroll.setStyleSheet(grey_scroll_style_sheet)

        # Create the view for the level
        self.create_table_level()

        # Create the view for the horizontal header
        self.create_table_header()

        # Create the view for the vertical index
        self.create_table_index()

        # Create the model and view of the data
        self.dataModel = DataFrameModel(data, parent=self)
        self.dataModel.dataChanged.connect(self.save_and_close_enable)
        # data frame view
        self.create_data_table()

        self.layout.addWidget(self.hscroll, 2, 0, 1, 2)
        self.layout.addWidget(self.vscroll, 0, 2, 2, 1)

        # autosize columns on-demand
        self._autosized_cols = set()
        # self._max_autosize_ms = None
        # this is the time limit for resizing
        self.setAutosizeLimit(0.0001)
        self.dataTable.installEventFilter(self)

        avg_width = self.fontMetrics().averageCharWidth()
        self.min_trunc = avg_width * 12  # Minimum size for columns
        self.max_width = avg_width * 64  # Maximum size for columns

        self.setLayout(self.layout)
        self.setMinimumSize(400, 300)
        # Make the dialog act as a window
        self.setWindowFlags(Qt.Window)
        btn_layout = QHBoxLayout()

        btn = QPushButton(_("Format"))
        # disable format button for int type
        btn_layout.addWidget(btn)
        btn.clicked.connect(self.change_format)
        # btn = QPushButton(_('Resize'))
        # btn_layout.addWidget(btn)
        # btn.clicked.connect(self.resize_to_contents)
        btn = QPushButton(_('Plot'))
        btn_layout.addWidget(btn)
        btn.clicked.connect(self.plot_selected_columns)

        btn = QPushButton(_('Reset Filter'))
        btn_layout.addWidget(btn)
        btn.clicked.connect(self.reset_filter)

        bgcolor = QCheckBox(_('Background color'))
        bgcolor.setChecked(self.dataModel.bgcolor_enabled)
        bgcolor.setEnabled(self.dataModel.bgcolor_enabled)
        bgcolor.stateChanged.connect(self.change_bgcolor_enable)
        btn_layout.addWidget(bgcolor)

        self.is_resized_by_contents_only = False
        resize_by_contents = QCheckBox(_('Resize by contents'))
        resize_by_contents.setChecked(self.is_resized_by_contents_only)
        resize_by_contents.stateChanged.connect(self.change_is_resized_by_contents)
        btn_layout.addWidget(resize_by_contents)

        '''
        self.bgcolor_global = QCheckBox(_('Column min/max'))
        self.bgcolor_global.setChecked(self.dataModel.colum_avg_enabled)
        self.bgcolor_global.setEnabled(not self.is_series and
                                       self.dataModel.bgcolor_enabled)
        self.bgcolor_global.stateChanged.connect(self.dataModel.colum_avg)
        btn_layout.addWidget(self.bgcolor_global)
        '''

        self.textbox = QLineEdit()
        btn_layout.addWidget(self.textbox)
        self.textbox.setPlaceholderText('Filtered command')

        btn_layout.addStretch()

        self.btn_save_and_close = QPushButton(_('Save and Close'))
        self.btn_save_and_close.setDisabled(True)
        self.btn_save_and_close.clicked.connect(self.accept)
        btn_layout.addWidget(self.btn_save_and_close)

        self.btn_close = QPushButton(_('Close'))
        self.btn_close.setAutoDefault(True)
        self.btn_close.setDefault(True)
        self.btn_close.clicked.connect(self.reject)
        btn_layout.addWidget(self.btn_close)

        btn_layout.setContentsMargins(4, 4, 4, 4)
        self.layout.addLayout(btn_layout, 4, 0, 1, 2)
        self.setModel(self.dataModel, relayout=True)
        self.resizeIndexColumnAtInitialization()
        QApplication.processEvents()
        return True

    @Slot(QModelIndex, QModelIndex)
    def save_and_close_enable(self, top_left, bottom_right):
        """Handle the data change event to enable the save and close button."""
        self.btn_save_and_close.setEnabled(True)
        self.btn_save_and_close.setAutoDefault(True)
        self.btn_save_and_close.setDefault(True)

    def create_table_level(self):
        """Create the QTableView that will hold the level model."""
        self.table_level = QTableView()
        self.table_level.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_level.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_level.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_level.setFrameStyle(QFrame.Plain)
        self.custom_label_view = CustomHeaderViewIndex(self.table_level)
        self.custom_label_view.setSectionsClickable(True)
        self.table_level.setHorizontalHeader(self.custom_label_view)
        self.table_level.horizontalHeader().sectionResized.connect(
            self._index_resized)
        self.table_level.verticalHeader().sectionResized.connect(
            self._header_resized)
        self.table_level.setItemDelegate(QItemDelegate())
        self.layout.addWidget(self.table_level, 0, 0)
        self.table_level.setContentsMargins(0, 0, 0, 0)
        self.table_level.horizontalHeader().sectionClicked.connect(
            self.sortByIndex)

    def create_table_header(self):
        """Create the QTableView that will hold the header model."""
        self.table_header = QTableView()
        # move all the scroll bar related settings here
        # set here as custom_header_view uses scrollbar in table_header
        self.table_header.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_header.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_header.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.table_header.setHorizontalScrollBar(self.hscroll)
        self.custom_header_view = CustomHeaderViewEditor(self.table_header)
        # needs to set cliackable manually,  not sure what else needs to self manually when creating qheaderview class
        # maybe check value of each field?
        # https://github.com/spyder-ide/qtpy/blob/master/qtpy/tests/test_patch_qheaderview.py
        self.custom_header_view.setSectionsClickable(True)
        self.table_header.setHorizontalHeader(self.custom_header_view)
        self.table_header.horizontalHeader().sectionResized.connect(self._column_resized)
        # the rest are defaults
        self.table_header.verticalHeader().hide()
        self.table_header.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_header.setFrameStyle(QFrame.Plain)
        self.table_header.setItemDelegate(QItemDelegate())
        self.layout.addWidget(self.table_header, 0, 1)

    def create_table_index(self):
        """Create the QTableView that will hold the index model."""
        self.table_index = QTableView()
        self.table_index.horizontalHeader().hide()
        self.table_index.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table_index.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_index.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table_index.setVerticalScrollMode(QTableView.ScrollPerPixel)
        self.table_index.setVerticalScrollBar(self.vscroll)
        self.table_index.setFrameStyle(QFrame.Plain)
        self.table_index.verticalHeader().sectionResized.connect(
            self._row_resized)
        self.table_index.setItemDelegate(QItemDelegate())
        self.layout.addWidget(self.table_index, 1, 0)
        self.table_index.setContentsMargins(0, 0, 0, 0)

    def create_data_table(self):
        """Create the QTableView that will hold the data model."""
        self.dataTable = DataFrameView(self, self.dataModel,
                                       self.table_header.horizontalHeader(),
                                       self.hscroll, self.vscroll)
        self.dataTable.verticalHeader().hide()
        self.dataTable.horizontalHeader().hide()
        self.dataTable.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dataTable.setVerticalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.dataTable.setHorizontalScrollMode(QTableView.ScrollPerPixel)
        self.dataTable.setVerticalScrollMode(QTableView.ScrollPerPixel)
        self.dataTable.setFrameStyle(QFrame.Plain)
        self.dataTable.setItemDelegate(QItemDelegate())
        self.layout.addWidget(self.dataTable, 1, 1)
        self.setFocusProxy(self.dataTable)
        self.dataTable.sig_sort_by_column.connect(self._sort_update)
        self.dataTable.sig_fetch_more_columns.connect(self._fetch_more_columns)
        self.dataTable.sig_fetch_more_rows.connect(self._fetch_more_rows)
        self.dataTable.sig_plot.connect(self.plot_selected_columns)
        self.dataTable.sig_subplot.connect(self.plot_selected_columns_subplot)
        self.dataTable.sig_reset_and_scroll_to.connect(self.reset_and_scroll_to)
        self.dataTable.sig_reset_sort_indicator.connect(self._reset_sort_indicator)

    def sortByIndex(self, index):
        """Implement a Index sort."""
        self.table_level.horizontalHeader().setSortIndicatorShown(True)
        sort_order = self.table_level.horizontalHeader().sortIndicatorOrder()
        self.table_index.model().sort(index, sort_order)
        self._sort_update()
        self.dataTable._reset_sort_indicator()

    def _reset_sort_indicator(self):
        self.table_level.horizontalHeader().setSortIndicatorShown(False)

    def model(self):
        """Get the model of the dataframe."""
        return self._model

    def _column_resized(self, col, old_width, new_width):
        """Update the column width."""
        self.dataTable.setColumnWidth(col, new_width)
        self._update_layout()

    def _row_resized(self, row, old_height, new_height):
        """Update the row height."""
        self.dataTable.setRowHeight(row, new_height)
        self._update_layout()

    def _index_resized(self, col, old_width, new_width):
        """Resize the corresponding column of the index section selected."""
        self.table_index.setColumnWidth(col, new_width)
        self._update_layout()

    def _header_resized(self, row, old_height, new_height):
        """Resize the corresponding row of the header section selected."""
        self.table_header.setRowHeight(row, new_height)
        self._update_layout()

    # the width and height of level, index and header are first set here
    # by using sizeHint on verticalHeader etc,
    # and then re-adjust from the contents in datatable
    def _update_layout(self):
        """Set the width and height of the QTableViews and hide rows."""
        h_width = max(self.table_level.verticalHeader().sizeHint().width(),
                      self.table_index.verticalHeader().sizeHint().width())
        self.table_level.verticalHeader().setFixedWidth(h_width)
        self.table_index.verticalHeader().setFixedWidth(h_width)

        # last_row >= 0 for non empty dataframe
        last_row = self._model.header_shape[0] - 1
        if last_row < 0:
            hdr_height = self.table_level.horizontalHeader().height()
        else:
            # hdr_height = self.table_level.rowViewportPosition(last_row) + \
            #              self.table_level.rowHeight(last_row) + \
            #              self.table_level.horizontalHeader().height()
            hdr_height = self.custom_header_view.sizeHint().height()

            # Check if the header shape has only one row (which display the
            # same info than the horizontal header).
            if last_row == 0:
                # This is to hide the empty data table of table_level and table_header?
                self.table_level.setRowHidden(0, True)
                self.table_header.setRowHidden(0, True)
        self.table_header.setFixedHeight(hdr_height)
        self.table_level.setFixedHeight(hdr_height)

        last_col = self._model.header_shape[1] - 1
        if last_col < 0:
            idx_width = self.table_level.verticalHeader().width()
        else:
            idx_width = self.table_level.columnViewportPosition(last_col) + \
                        self.table_level.columnWidth(last_col) + \
                        self.table_level.verticalHeader().width()
        self.table_index.setFixedWidth(idx_width)
        self.table_level.setFixedWidth(idx_width)
        self._resizeAllColumnsToContents()

    def _reset_model(self, table, model):
        """Set the model in the given table."""
        old_sel_model = table.selectionModel()
        table.setModel(model)
        if old_sel_model:
            del old_sel_model

    def setAutosizeLimit(self, limit_ms):
        """Set maximum size for columns."""
        self._max_autosize_ms = limit_ms

    def setModel(self, model, relayout=True):
        """Set the model for the data, header/index and level views."""
        self._model = model
        # The original behavior is to resize column only when selected (clicked)
        # We have changed this to resize when in view
        # sel_model = self.dataTable.selectionModel()
        # sel_model.currentColumnChanged.connect(
        #     self._resizeCurrentColumnToContents)

        # Asociate the models (level, vertical index and horizontal header)
        # with its corresponding view.
        self._reset_model(self.table_level, DataFrameLevelModel(model,
                                                                self.palette(),
                                                                self.font()))
        self._reset_model(self.table_header, DataFrameHeaderModel(
            model,
            0,
            self.palette()))
        self._reset_model(self.table_index, DataFrameHeaderModel(
            model,
            1,
            self.palette()))

        # setting the custom header
        if self._model.shape[1] > 0:
            self.custom_header_view.setFilterBoxes(self._model.shape[1])
            self.custom_label_view.setQLabel(str(self._model.shape[0]))
        # set signal once only
        if self.custom_header_view.receivers(self.custom_header_view.filterActivated) == 0:
            self.custom_header_view.filterActivated.connect(self.handleFilterActivated)

        # Needs to be called after setting all table models
        if relayout:
            self._update_layout()

    def setCurrentIndex(self, y, x):
        """Set current selection."""
        self.dataTable.selectionModel().setCurrentIndex(
            self.dataTable.model().index(y, x),
            QItemSelectionModel.ClearAndSelect)

    def _sizeHintForColumn(self, table, col, limit_ms=None):
        """Get the size hint for a given column in a table."""
        """
        This function tries to find the maximum width for each column, by checking the width of each cell
        there is a time limit for this function, default is None
        Note that the data does not always equal the full dataframe (depends on the data loaded to the table view).
        table: QTableView
        col: int
        """
        max_row = table.model().rowCount()
        lm_start = time.clock()
        lm_row = 64 if limit_ms else max_row
        max_width = self.min_trunc
        for row in range(max_row):
            v = table.sizeHintForIndex(table.model().index(row, col))
            max_width = max(max_width, v.width())
            if row > lm_row:
                lm_now = time.clock()
                lm_elapsed = (lm_now - lm_start) * 1000
                if lm_elapsed >= limit_ms:
                    break
                lm_row = int((row / lm_elapsed) * limit_ms)
        return max_width

    def _resizeColumnToContents(self, header, data, col, limit_ms):
        """Resize a column width by width of header and data
        hdr_width should be 0 as header.model() should be empty (when no multi index)
        check the size of the name in header if is_resized_by_contents
        """
        hdr_width = self._sizeHintForColumn(header, col, limit_ms)
        hdr_name_width = header.horizontalHeader().sectionSizeHint(col)
        data_width = self._sizeHintForColumn(data, col, limit_ms)
        if self.is_resized_by_contents_only:
            width = min(self.max_width, max(hdr_width, data_width))
        else:
            width = min(self.max_width, max(hdr_width, hdr_name_width, data_width))
        header.setColumnWidth(col, width)

    '''
    def _resizeColumnsToContents(self, header, data, limit_ms):
        """Resize all the columns in data"""
        max_col = data.model().columnCount()
        if limit_ms is None:
            max_col_ms = None
        else:
            max_col_ms = limit_ms / max(1, max_col)
        for col in range(max_col):
            self._resizeColumnToContents(header, data, col, max_col_ms)
    '''

    def eventFilter(self, obj, event):
        """Override eventFilter to catch resize event."""
        # from installeventfilter, dataframeeditor receive all events from dataTable(DataFrameView)
        # the events received are catched by this function and only act on it when it is resize
        if obj == self.dataTable and event.type() == QEvent.Resize:
            # Tried calling _update_layout() before
            # which would reset the width/height of header and index
            # Now follows the original implementation
            # self._update_layout()
            self._resizeAllColumnsToContents()
        return False

    def keyPressEvent(self, event):
        # This is to block the enter/return key to its parent and exiting the application
        if event.key() == Qt.Key_Enter or event.key() == Qt.Key_Return:
            return
        else:
            super(DataFrameEditor, self).keyPressEvent(event)

    def _resizeAllColumnsToContents(self):
        """Resize all columns that is not already resized (checked with _autosized_cols) """
        # decide starting column, ending column and time
        index_column = self.dataTable.rect().topLeft().x()
        start = col = self.dataTable.columnAt(index_column)
        width = self._model.shape[1]
        # end = self.dataTable.columnAt(self.dataTable.rect().bottomRight().x())
        end = width
        # end = width if end == -1 else end + 1
        if self._max_autosize_ms is None:
            max_col_ms = None
        else:
            max_col_ms = self._max_autosize_ms / max(1, end - start)
        while col < end:
            # resized = False
            if col not in self._autosized_cols:
                self._autosized_cols.add(col)
                resized = True
                self._resizeColumnToContents(self.table_header, self.dataTable,
                                             col, max_col_ms)
            col += 1
            # if resized:
            #     # As we resize columns, the boundary will change
            #     index_column = self.dataTable.rect().bottomRight().x()
            #     end = self.dataTable.columnAt(index_column)
            #     end = width if end == -1 else end + 1
            #     # logger.info("Update end to:" + str(end))
            #     if max_col_ms is not None:
            #         max_col_ms = self._max_autosize_ms / max(1, end - start)

    '''
    def _resizeCurrentColumnToContents(self, new_index, old_index):
        """Resize the current column to its contents."""
        if new_index.column() not in self._autosized_cols:
            # Ensure the requested column is fully into view after resizing
            self._resizeAllColumnsToContents()
            self.dataTable.scrollTo(new_index)
    '''

    '''This is called by table index only, unused
    def resizeColumnsToContents(self):
        """Resize the columns to its contents."""
        self._autosized_cols = set()
        self._resizeColumnsToContents(self.table_level,
                                      self.table_index, self._max_autosize_ms)
        self._update_layout()
        self.table_level.resizeColumnsToContents()
    # '''

    def resizeIndexColumnAtInitialization(self):
        """call resize index, this should be the only place where index width is resized
        1. resize by its contents
        2. resize of by max of width and the custom header width
        """
        self._resizeColumnToContents(self.table_level, self.table_index, 0, self._max_autosize_ms)
        width = max(self.table_level.columnWidth(0), self.table_level.horizontalHeader().sizeHint().width())
        self.table_level.setColumnWidth(0, width)

    def change_bgcolor_enable(self, state):
        """
        This is implementet so column min/max is only active when bgcolor is
        """
        self.dataModel.bgcolor(state)
        self.bgcolor_global.setEnabled(not self.is_series and state > 0)

    def change_is_resized_by_contents(self, state):
        self.is_resized_by_contents_only = state > 0
        self._autosized_cols = set()
        self._resizeAllColumnsToContents()

    def change_format(self):
        """
        Ask user for display format for floats and use it.

        This function also checks whether the format is valid and emits
        `sig_option_changed`.
        """
        format, valid = QInputDialog.getText(self, _('Format'),
                                             _("Float formatting"),
                                             QLineEdit.Normal,
                                             self.dataModel.get_format())
        if valid:
            format = str(format)
            try:
                format % 1.1
            except:
                msg = _("Format ({}) is incorrect").format(format)
                QMessageBox.critical(self, _("Error"), msg)
                return
            if not format.startswith('%'):
                msg = _("Format ({}) should start with '%'").format(format)
                QMessageBox.critical(self, _("Error"), msg)
                return
            self.dataModel.set_format(format)
            self.sig_option_changed.emit('dataframe_format', format)

    def get_value(self):
        """Return modified Dataframe -- this is *not* a copy"""
        # It is import to avoid accessing Qt C++ object as it has probably
        # already been destroyed, due to the Qt.WA_DeleteOnClose attribute
        df = self.dataModel.get_data()
        if self.is_series:
            return df.iloc[:, 0]
        else:
            return df

    '''
    def _update_header_size(self):
        """Update the column width of the header."""
        column_count = self.table_header.model().columnCount()
        for index in range(0, column_count):
            if index < column_count:
                column_width = self.dataTable.columnWidth(index)
                self.table_header.setColumnWidth(index, column_width)
            else:
                break
    '''

    def _sort_update(self):
        """
        Update the model for all the QTableView objects.

        Uses the model of the dataTable as the base.
        """
        self.setModel(self.dataTable.model())

    def _fetch_more_columns(self):
        """Fetch more data for the header (columns)."""
        self.table_header.model().fetch_more()

    def _fetch_more_rows(self):
        """Fetch more data for the index (rows)."""
        self.table_index.model().fetch_more()

    '''
    def resize_to_contents(self):
        QApplication.setOverrideCursor(QCursor(Qt.WaitCursor))
        self.dataTable.resizeColumnsToContents()
        self.dataModel.fetch_more(columns=True)
        self.dataTable.resizeColumnsToContents()
        self._update_header_size()
        QApplication.restoreOverrideCursor()
    '''

    def set_filter(self, filter_list):
        self.dataModel.set_filter(filter_list, self.df_name)
        # reset size of other views
        self.setModel(self.dataTable.model(), relayout=False)
        # sort with the order before filtering
        if self.dataTable.sort_old != [None]:
            self.dataModel.sort(*self.dataTable.sort_old)

    def plot_selected_columns(self, is_subplot=False):
        # check selected Index
        selected_idx = set([x.column() for x in self.dataTable.selectedIndexes()])
        df = self.dataModel.df
        plot_list = []
        for idx, col in enumerate(df):
            # filter non numbers
            if df[col].dtype not in REAL_NUMBER_TYPES + COMPLEX_NUMBER_TYPES:
                continue
            # filter not selected
            if selected_idx and idx not in selected_idx:
                continue
            plot_list.append((df[col], df.columns[idx]))
        # import plotting libraries
        try:
            import plotly
            use_plotly = True
        except ImportError:
            use_plotly = False
        if use_plotly:
            from plotly.offline import plot
            import plotly.graph_objs as go
            trace_list = []
            for y, name in plot_list:
                trace_list.append(go.Scatter(x=df.index, y=y, name=name))
            fig = go.Figure(data=trace_list)
            plot(fig)
        else:
            import spyder.pyplot as plt
            plt.figure()
            is_anything_plotted = False
            plot_count = 1
            for y, name in plot_list:
                try:
                    if not is_subplot:
                        plt.plot(y, label=name, alpha=0.6)
                    else:
                        plt.subplot(len(selected_idx), 1, plot_count)
                        plt.plot(y)
                        plt.legend(loc='upper right')
                        plot_count += 1
                    is_anything_plotted = True
                except ValueError:
                    pass
            if not is_anything_plotted:
                plt.close()
            else:
                plt.xticks(rotation=90)
                plt.tight_layout()
                plt.legend()
                plt.show()

    def plot_selected_columns_subplot(self):
        self.plot_selected_columns(is_subplot=True)

    def handleFilterActivated(self):
        self.set_filter(self.custom_header_view.getText())
        self.textbox.setText(self.dataModel.filtered_text)

    def reset_filter(self):
        self.custom_header_view._clearFilterText()
        self.handleFilterActivated()

    def reset_and_scroll_to(self):
        selected_row_list = set([x.row() for x in self.dataTable.selectedIndexes()])
        if len(selected_row_list) > 1:
            QMessageBox.critical(self, "Error", "Only one row should be selected.")
            return
        if self.dataModel.idx_tracker is not None:
            picked_row_rel_to_original_df = self.dataModel.idx_tracker.iloc[next(iter(selected_row_list))]
            self.reset_filter()
            self.dataTable.scroll_to_and_select(picked_row_rel_to_original_df, 0)
        # check index
        # non unique if not integer
        # TODO warning if more than 1 column selected


# ==============================================================================
# Tests
# ==============================================================================


def _test_edit(data, title="", parent=None):
    """Test subroutine"""
    app = qapplication()  # analysis:ignore
    # cross platform pyqt5 style
    app.setStyle("Fusion")
    dlg = DataFrameEditor(parent=parent)

    if dlg.setup_and_check(data, title=title):
        dlg.show()  # need this for plot to be able to close, like collectionseditor
        dlg.exec_()
        return dlg.get_value()
    else:
        import sys
        sys.exit(1)


import cProfile


def _test_wrapper(func, df, is_profiling=False):
    if is_profiling:
        pr = cProfile.Profile()
        pr.enable()
        func(df)
        pr.disable()
        # after your program ends
        pr.print_stats(sort="calls")
    else:
        func(df)


def test():
    """DataFrame editor test"""
    import random
    import datetime
    string_list = ['AAAA', 'BBBBB', 'CCCCCCC', 'DDDDDDDD', 'EEEEEEE']
    string_list_2 = ['AAAA', 'BBBBB', np.nan]
    variety_list = ['AAAA', 1, np.nan]
    datetime_list = [datetime.datetime(2018, 1, 1, 1, 1, 1), datetime.datetime(2022, 2, 2, 2, 2, 2)]
    true_false_list = [True, False]
    float_inf_list = [0.11, 999.8, np.inf, -np.inf]
    nrow = 10000
    r = random.Random(502)
    df1 = DataFrame([r.choice(string_list) for _ in range(nrow)], columns=['Test'])
    df1['num'] = range(nrow)
    df1.loc[8, 'Test2'] = float('nan')
    df1['Test3'] = df1['Test']
    df1['really_long_column_name_1'] = df1['Test']
    df1['really_long_column_name_2'] = df1['Test']
    df1['Test6'] = "Test"
    df1['Test7'] = 5
    df1 = df1.join([DataFrame([r.choice(variety_list) for _ in range(nrow)], columns=['variety_list'])])
    df1 = df1.join([DataFrame([r.choice(true_false_list) for _ in range(nrow)], columns=['true_false_list'])])
    df1 = df1.join([DataFrame([r.choice(string_list_2) for _ in range(nrow)], columns=['string_list_2'])])
    df1 = df1.join([DataFrame([r.choice(datetime_list) for _ in range(nrow)], columns=['date_time'])])
    df1 = df1.join([DataFrame([r.choice(float_inf_list) for _ in range(nrow)], columns=['float_inf_list'])])
    df1 = df1.join([DataFrame(np.random.rand(nrow, 10), columns=list(map(chr, range(97, 107))))])
    df1.loc[1, 'a'] = float('nan')
    df1 = df1.join([DataFrame(np.random.rand(nrow, 5) * 20, columns=['A', 'B', 'C', 'D', 'E'])])
    df1 = df1.join([DataFrame([{'F': [1, 2, 3, 4]}])])
    df1['super_super_super_unacceptable_long_column_name_that_should_not_happen'] = df1['Test']
    df1.set_index('date_time', inplace=True)

    _test_wrapper(_test_edit, df1, is_profiling=False)
    # from pandas import MultiIndex
    # import numpy
    #
    # arrays = [numpy.array(['bar', 'bar', 'baz', 'baz',
    #                        'foo', 'foo', 'qux', 'qux']),
    #           numpy.array(['one', 'two', 'one', 'two',
    #                        'one', 'two', 'one', 'two'])]
    # tuples = list(zip(*arrays))
    # index = MultiIndex.from_tuples(tuples, names=['first', 'second'])
    # df = DataFrame(numpy.random.randn(6, 6), index=index[:6],
    #               columns=index[:6])
    # _test_wrapper(_test_edit, df, is_profiling=False)


if __name__ == '__main__':
    test()
