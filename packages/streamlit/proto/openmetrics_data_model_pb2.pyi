"""
@generated by mypy-protobuf.  Do not edit manually!
isort:skip_file
The OpenMetrics protobuf schema which defines the protobuf wire format. 
Ensure to interpret "required" as semantically required for a valid message.
All string fields MUST be UTF-8 encoded strings.
"""

import builtins
import collections.abc
import google.protobuf.descriptor
import google.protobuf.internal.containers
import google.protobuf.internal.enum_type_wrapper
import google.protobuf.message
import google.protobuf.timestamp_pb2
import sys
import typing

if sys.version_info >= (3, 10):
    import typing as typing_extensions
else:
    import typing_extensions

DESCRIPTOR: google.protobuf.descriptor.FileDescriptor

class _MetricType:
    ValueType = typing.NewType("ValueType", builtins.int)
    V: typing_extensions.TypeAlias = ValueType

class _MetricTypeEnumTypeWrapper(google.protobuf.internal.enum_type_wrapper._EnumTypeWrapper[_MetricType.ValueType], builtins.type):
    DESCRIPTOR: google.protobuf.descriptor.EnumDescriptor
    UNKNOWN: _MetricType.ValueType  # 0
    """Unknown must use unknown MetricPoint values."""
    GAUGE: _MetricType.ValueType  # 1
    """Gauge must use gauge MetricPoint values."""
    COUNTER: _MetricType.ValueType  # 2
    """Counter must use counter MetricPoint values."""
    STATE_SET: _MetricType.ValueType  # 3
    """State set must use state set MetricPoint values."""
    INFO: _MetricType.ValueType  # 4
    """Info must use info MetricPoint values."""
    HISTOGRAM: _MetricType.ValueType  # 5
    """Histogram must use histogram value MetricPoint values."""
    GAUGE_HISTOGRAM: _MetricType.ValueType  # 6
    """Gauge histogram must use histogram value MetricPoint values."""
    SUMMARY: _MetricType.ValueType  # 7
    """Summary quantiles must use summary value MetricPoint values."""

class MetricType(_MetricType, metaclass=_MetricTypeEnumTypeWrapper):
    """The type of a Metric."""

UNKNOWN: MetricType.ValueType  # 0
"""Unknown must use unknown MetricPoint values."""
GAUGE: MetricType.ValueType  # 1
"""Gauge must use gauge MetricPoint values."""
COUNTER: MetricType.ValueType  # 2
"""Counter must use counter MetricPoint values."""
STATE_SET: MetricType.ValueType  # 3
"""State set must use state set MetricPoint values."""
INFO: MetricType.ValueType  # 4
"""Info must use info MetricPoint values."""
HISTOGRAM: MetricType.ValueType  # 5
"""Histogram must use histogram value MetricPoint values."""
GAUGE_HISTOGRAM: MetricType.ValueType  # 6
"""Gauge histogram must use histogram value MetricPoint values."""
SUMMARY: MetricType.ValueType  # 7
"""Summary quantiles must use summary value MetricPoint values."""
global___MetricType = MetricType

@typing.final
class MetricSet(google.protobuf.message.Message):
    """The top-level container type that is encoded and sent over the wire."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    METRIC_FAMILIES_FIELD_NUMBER: builtins.int
    @property
    def metric_families(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MetricFamily]:
        """Each MetricFamily has one or more MetricPoints for a single Metric."""

    def __init__(
        self,
        *,
        metric_families: collections.abc.Iterable[global___MetricFamily] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["metric_families", b"metric_families"]) -> None: ...

global___MetricSet = MetricSet

@typing.final
class MetricFamily(google.protobuf.message.Message):
    """One or more Metrics for a single MetricFamily, where each Metric
    has one or more MetricPoints.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    TYPE_FIELD_NUMBER: builtins.int
    UNIT_FIELD_NUMBER: builtins.int
    HELP_FIELD_NUMBER: builtins.int
    METRICS_FIELD_NUMBER: builtins.int
    name: builtins.str
    """Required."""
    type: global___MetricType.ValueType
    """Optional."""
    unit: builtins.str
    """Optional."""
    help: builtins.str
    """Optional."""
    @property
    def metrics(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Metric]:
        """Optional."""

    def __init__(
        self,
        *,
        name: builtins.str = ...,
        type: global___MetricType.ValueType = ...,
        unit: builtins.str = ...,
        help: builtins.str = ...,
        metrics: collections.abc.Iterable[global___Metric] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["help", b"help", "metrics", b"metrics", "name", b"name", "type", b"type", "unit", b"unit"]) -> None: ...

global___MetricFamily = MetricFamily

@typing.final
class Metric(google.protobuf.message.Message):
    """A single metric with a unique set of labels within a metric family."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    LABELS_FIELD_NUMBER: builtins.int
    METRIC_POINTS_FIELD_NUMBER: builtins.int
    @property
    def labels(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Label]:
        """Optional."""

    @property
    def metric_points(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___MetricPoint]:
        """Optional."""

    def __init__(
        self,
        *,
        labels: collections.abc.Iterable[global___Label] | None = ...,
        metric_points: collections.abc.Iterable[global___MetricPoint] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["labels", b"labels", "metric_points", b"metric_points"]) -> None: ...

global___Metric = Metric

@typing.final
class Label(google.protobuf.message.Message):
    """A name-value pair. These are used in multiple places: identifying
    timeseries, value of INFO metrics, and exemplars in Histograms.
    """

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    NAME_FIELD_NUMBER: builtins.int
    VALUE_FIELD_NUMBER: builtins.int
    name: builtins.str
    """Required."""
    value: builtins.str
    """Required."""
    def __init__(
        self,
        *,
        name: builtins.str = ...,
        value: builtins.str = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["name", b"name", "value", b"value"]) -> None: ...

global___Label = Label

@typing.final
class MetricPoint(google.protobuf.message.Message):
    """A MetricPoint in a Metric."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    UNKNOWN_VALUE_FIELD_NUMBER: builtins.int
    GAUGE_VALUE_FIELD_NUMBER: builtins.int
    COUNTER_VALUE_FIELD_NUMBER: builtins.int
    HISTOGRAM_VALUE_FIELD_NUMBER: builtins.int
    STATE_SET_VALUE_FIELD_NUMBER: builtins.int
    INFO_VALUE_FIELD_NUMBER: builtins.int
    SUMMARY_VALUE_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    @property
    def unknown_value(self) -> global___UnknownValue: ...
    @property
    def gauge_value(self) -> global___GaugeValue: ...
    @property
    def counter_value(self) -> global___CounterValue: ...
    @property
    def histogram_value(self) -> global___HistogramValue: ...
    @property
    def state_set_value(self) -> global___StateSetValue: ...
    @property
    def info_value(self) -> global___InfoValue: ...
    @property
    def summary_value(self) -> global___SummaryValue: ...
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Optional."""

    def __init__(
        self,
        *,
        unknown_value: global___UnknownValue | None = ...,
        gauge_value: global___GaugeValue | None = ...,
        counter_value: global___CounterValue | None = ...,
        histogram_value: global___HistogramValue | None = ...,
        state_set_value: global___StateSetValue | None = ...,
        info_value: global___InfoValue | None = ...,
        summary_value: global___SummaryValue | None = ...,
        timestamp: google.protobuf.timestamp_pb2.Timestamp | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["counter_value", b"counter_value", "gauge_value", b"gauge_value", "histogram_value", b"histogram_value", "info_value", b"info_value", "state_set_value", b"state_set_value", "summary_value", b"summary_value", "timestamp", b"timestamp", "unknown_value", b"unknown_value", "value", b"value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["counter_value", b"counter_value", "gauge_value", b"gauge_value", "histogram_value", b"histogram_value", "info_value", b"info_value", "state_set_value", b"state_set_value", "summary_value", b"summary_value", "timestamp", b"timestamp", "unknown_value", b"unknown_value", "value", b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["value", b"value"]) -> typing.Literal["unknown_value", "gauge_value", "counter_value", "histogram_value", "state_set_value", "info_value", "summary_value"] | None: ...

global___MetricPoint = MetricPoint

@typing.final
class UnknownValue(google.protobuf.message.Message):
    """Value for UNKNOWN MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DOUBLE_VALUE_FIELD_NUMBER: builtins.int
    INT_VALUE_FIELD_NUMBER: builtins.int
    double_value: builtins.float
    int_value: builtins.int
    def __init__(
        self,
        *,
        double_value: builtins.float = ...,
        int_value: builtins.int = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["double_value", b"double_value", "int_value", b"int_value", "value", b"value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["double_value", b"double_value", "int_value", b"int_value", "value", b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["value", b"value"]) -> typing.Literal["double_value", "int_value"] | None: ...

global___UnknownValue = UnknownValue

@typing.final
class GaugeValue(google.protobuf.message.Message):
    """Value for GAUGE MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DOUBLE_VALUE_FIELD_NUMBER: builtins.int
    INT_VALUE_FIELD_NUMBER: builtins.int
    double_value: builtins.float
    int_value: builtins.int
    def __init__(
        self,
        *,
        double_value: builtins.float = ...,
        int_value: builtins.int = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["double_value", b"double_value", "int_value", b"int_value", "value", b"value"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["double_value", b"double_value", "int_value", b"int_value", "value", b"value"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["value", b"value"]) -> typing.Literal["double_value", "int_value"] | None: ...

global___GaugeValue = GaugeValue

@typing.final
class CounterValue(google.protobuf.message.Message):
    """Value for COUNTER MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    DOUBLE_VALUE_FIELD_NUMBER: builtins.int
    INT_VALUE_FIELD_NUMBER: builtins.int
    CREATED_FIELD_NUMBER: builtins.int
    EXEMPLAR_FIELD_NUMBER: builtins.int
    double_value: builtins.float
    int_value: builtins.int
    @property
    def created(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """The time values began being collected for this counter.
        Optional.
        """

    @property
    def exemplar(self) -> global___Exemplar:
        """Optional."""

    def __init__(
        self,
        *,
        double_value: builtins.float = ...,
        int_value: builtins.int = ...,
        created: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        exemplar: global___Exemplar | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["created", b"created", "double_value", b"double_value", "exemplar", b"exemplar", "int_value", b"int_value", "total", b"total"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["created", b"created", "double_value", b"double_value", "exemplar", b"exemplar", "int_value", b"int_value", "total", b"total"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["total", b"total"]) -> typing.Literal["double_value", "int_value"] | None: ...

global___CounterValue = CounterValue

@typing.final
class HistogramValue(google.protobuf.message.Message):
    """Value for HISTOGRAM or GAUGE_HISTOGRAM MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class Bucket(google.protobuf.message.Message):
        """Bucket is the number of values for a bucket in the histogram
        with an optional exemplar.
        """

        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        COUNT_FIELD_NUMBER: builtins.int
        UPPER_BOUND_FIELD_NUMBER: builtins.int
        EXEMPLAR_FIELD_NUMBER: builtins.int
        count: builtins.int
        """Required."""
        upper_bound: builtins.float
        """Optional."""
        @property
        def exemplar(self) -> global___Exemplar:
            """Optional."""

        def __init__(
            self,
            *,
            count: builtins.int = ...,
            upper_bound: builtins.float = ...,
            exemplar: global___Exemplar | None = ...,
        ) -> None: ...
        def HasField(self, field_name: typing.Literal["exemplar", b"exemplar"]) -> builtins.bool: ...
        def ClearField(self, field_name: typing.Literal["count", b"count", "exemplar", b"exemplar", "upper_bound", b"upper_bound"]) -> None: ...

    DOUBLE_VALUE_FIELD_NUMBER: builtins.int
    INT_VALUE_FIELD_NUMBER: builtins.int
    COUNT_FIELD_NUMBER: builtins.int
    CREATED_FIELD_NUMBER: builtins.int
    BUCKETS_FIELD_NUMBER: builtins.int
    double_value: builtins.float
    int_value: builtins.int
    count: builtins.int
    """Optional."""
    @property
    def created(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """The time values began being collected for this histogram.
        Optional.
        """

    @property
    def buckets(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___HistogramValue.Bucket]:
        """Optional."""

    def __init__(
        self,
        *,
        double_value: builtins.float = ...,
        int_value: builtins.int = ...,
        count: builtins.int = ...,
        created: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        buckets: collections.abc.Iterable[global___HistogramValue.Bucket] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["created", b"created", "double_value", b"double_value", "int_value", b"int_value", "sum", b"sum"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["buckets", b"buckets", "count", b"count", "created", b"created", "double_value", b"double_value", "int_value", b"int_value", "sum", b"sum"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["sum", b"sum"]) -> typing.Literal["double_value", "int_value"] | None: ...

global___HistogramValue = HistogramValue

@typing.final
class Exemplar(google.protobuf.message.Message):
    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    VALUE_FIELD_NUMBER: builtins.int
    TIMESTAMP_FIELD_NUMBER: builtins.int
    LABEL_FIELD_NUMBER: builtins.int
    value: builtins.float
    """Required."""
    @property
    def timestamp(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """Optional."""

    @property
    def label(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Label]:
        """Labels are additional information about the exemplar value (e.g. trace id).
        Optional.
        """

    def __init__(
        self,
        *,
        value: builtins.float = ...,
        timestamp: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        label: collections.abc.Iterable[global___Label] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["timestamp", b"timestamp"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["label", b"label", "timestamp", b"timestamp", "value", b"value"]) -> None: ...

global___Exemplar = Exemplar

@typing.final
class StateSetValue(google.protobuf.message.Message):
    """Value for STATE_SET MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class State(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        ENABLED_FIELD_NUMBER: builtins.int
        NAME_FIELD_NUMBER: builtins.int
        enabled: builtins.bool
        """Required."""
        name: builtins.str
        """Required."""
        def __init__(
            self,
            *,
            enabled: builtins.bool = ...,
            name: builtins.str = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["enabled", b"enabled", "name", b"name"]) -> None: ...

    STATES_FIELD_NUMBER: builtins.int
    @property
    def states(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___StateSetValue.State]:
        """Optional."""

    def __init__(
        self,
        *,
        states: collections.abc.Iterable[global___StateSetValue.State] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["states", b"states"]) -> None: ...

global___StateSetValue = StateSetValue

@typing.final
class InfoValue(google.protobuf.message.Message):
    """Value for INFO MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    INFO_FIELD_NUMBER: builtins.int
    @property
    def info(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___Label]:
        """Optional."""

    def __init__(
        self,
        *,
        info: collections.abc.Iterable[global___Label] | None = ...,
    ) -> None: ...
    def ClearField(self, field_name: typing.Literal["info", b"info"]) -> None: ...

global___InfoValue = InfoValue

@typing.final
class SummaryValue(google.protobuf.message.Message):
    """Value for SUMMARY MetricPoint."""

    DESCRIPTOR: google.protobuf.descriptor.Descriptor

    @typing.final
    class Quantile(google.protobuf.message.Message):
        DESCRIPTOR: google.protobuf.descriptor.Descriptor

        QUANTILE_FIELD_NUMBER: builtins.int
        VALUE_FIELD_NUMBER: builtins.int
        quantile: builtins.float
        """Required."""
        value: builtins.float
        """Required."""
        def __init__(
            self,
            *,
            quantile: builtins.float = ...,
            value: builtins.float = ...,
        ) -> None: ...
        def ClearField(self, field_name: typing.Literal["quantile", b"quantile", "value", b"value"]) -> None: ...

    DOUBLE_VALUE_FIELD_NUMBER: builtins.int
    INT_VALUE_FIELD_NUMBER: builtins.int
    COUNT_FIELD_NUMBER: builtins.int
    CREATED_FIELD_NUMBER: builtins.int
    QUANTILE_FIELD_NUMBER: builtins.int
    double_value: builtins.float
    int_value: builtins.int
    count: builtins.int
    """Optional."""
    @property
    def created(self) -> google.protobuf.timestamp_pb2.Timestamp:
        """The time sum and count values began being collected for this summary.
        Optional.
        """

    @property
    def quantile(self) -> google.protobuf.internal.containers.RepeatedCompositeFieldContainer[global___SummaryValue.Quantile]:
        """Optional."""

    def __init__(
        self,
        *,
        double_value: builtins.float = ...,
        int_value: builtins.int = ...,
        count: builtins.int = ...,
        created: google.protobuf.timestamp_pb2.Timestamp | None = ...,
        quantile: collections.abc.Iterable[global___SummaryValue.Quantile] | None = ...,
    ) -> None: ...
    def HasField(self, field_name: typing.Literal["created", b"created", "double_value", b"double_value", "int_value", b"int_value", "sum", b"sum"]) -> builtins.bool: ...
    def ClearField(self, field_name: typing.Literal["count", b"count", "created", b"created", "double_value", b"double_value", "int_value", b"int_value", "quantile", b"quantile", "sum", b"sum"]) -> None: ...
    def WhichOneof(self, oneof_group: typing.Literal["sum", b"sum"]) -> typing.Literal["double_value", "int_value"] | None: ...

global___SummaryValue = SummaryValue