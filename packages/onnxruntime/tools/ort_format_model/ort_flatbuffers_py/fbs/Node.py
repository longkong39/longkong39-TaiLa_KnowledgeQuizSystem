# automatically generated by the FlatBuffers compiler, do not modify

# namespace: fbs

import flatbuffers
from flatbuffers.compat import import_numpy
np = import_numpy()

class Node(object):
    __slots__ = ['_tab']

    @classmethod
    def GetRootAs(cls, buf, offset=0):
        n = flatbuffers.encode.Get(flatbuffers.packer.uoffset, buf, offset)
        x = Node()
        x.Init(buf, n + offset)
        return x

    @classmethod
    def GetRootAsNode(cls, buf, offset=0):
        """This method is deprecated. Please switch to GetRootAs."""
        return cls.GetRootAs(buf, offset)
    @classmethod
    def NodeBufferHasIdentifier(cls, buf, offset, size_prefixed=False):
        return flatbuffers.util.BufferHasIdentifier(buf, offset, b"\x4F\x52\x54\x4D", size_prefixed=size_prefixed)

    # Node
    def Init(self, buf, pos):
        self._tab = flatbuffers.table.Table(buf, pos)

    # Node
    def Name(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(4))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def DocString(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(6))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def Domain(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(8))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def SinceVersion(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(10))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Node
    def Index(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(12))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Uint32Flags, o + self._tab.Pos)
        return 0

    # Node
    def OpType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(14))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def Type(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(16))
        if o != 0:
            return self._tab.Get(flatbuffers.number_types.Int32Flags, o + self._tab.Pos)
        return 0

    # Node
    def ExecutionProviderType(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(18))
        if o != 0:
            return self._tab.String(o + self._tab.Pos)
        return None

    # Node
    def Inputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Node
    def InputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def InputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(20))
        return o == 0

    # Node
    def Outputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Node
    def OutputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def OutputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(22))
        return o == 0

    # Node
    def Attributes(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            x = self._tab.Vector(o)
            x += flatbuffers.number_types.UOffsetTFlags.py_type(j) * 4
            x = self._tab.Indirect(x)
            from ort_flatbuffers_py.fbs.Attribute import Attribute
            obj = Attribute()
            obj.Init(self._tab.Bytes, x)
            return obj
        return None

    # Node
    def AttributesLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def AttributesIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(24))
        return o == 0

    # Node
    def InputArgCounts(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.Get(flatbuffers.number_types.Int32Flags, a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return 0

    # Node
    def InputArgCountsAsNumpy(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.GetVectorAsNumpy(flatbuffers.number_types.Int32Flags, o)
        return 0

    # Node
    def InputArgCountsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def InputArgCountsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(26))
        return o == 0

    # Node
    def ImplicitInputs(self, j):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            a = self._tab.Vector(o)
            return self._tab.String(a + flatbuffers.number_types.UOffsetTFlags.py_type(j * 4))
        return ""

    # Node
    def ImplicitInputsLength(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        if o != 0:
            return self._tab.VectorLen(o)
        return 0

    # Node
    def ImplicitInputsIsNone(self):
        o = flatbuffers.number_types.UOffsetTFlags.py_type(self._tab.Offset(28))
        return o == 0

def NodeStart(builder):
    builder.StartObject(13)

def Start(builder):
    NodeStart(builder)

def NodeAddName(builder, name):
    builder.PrependUOffsetTRelativeSlot(0, flatbuffers.number_types.UOffsetTFlags.py_type(name), 0)

def AddName(builder, name):
    NodeAddName(builder, name)

def NodeAddDocString(builder, docString):
    builder.PrependUOffsetTRelativeSlot(1, flatbuffers.number_types.UOffsetTFlags.py_type(docString), 0)

def AddDocString(builder, docString):
    NodeAddDocString(builder, docString)

def NodeAddDomain(builder, domain):
    builder.PrependUOffsetTRelativeSlot(2, flatbuffers.number_types.UOffsetTFlags.py_type(domain), 0)

def AddDomain(builder, domain):
    NodeAddDomain(builder, domain)

def NodeAddSinceVersion(builder, sinceVersion):
    builder.PrependInt32Slot(3, sinceVersion, 0)

def AddSinceVersion(builder, sinceVersion):
    NodeAddSinceVersion(builder, sinceVersion)

def NodeAddIndex(builder, index):
    builder.PrependUint32Slot(4, index, 0)

def AddIndex(builder, index):
    NodeAddIndex(builder, index)

def NodeAddOpType(builder, opType):
    builder.PrependUOffsetTRelativeSlot(5, flatbuffers.number_types.UOffsetTFlags.py_type(opType), 0)

def AddOpType(builder, opType):
    NodeAddOpType(builder, opType)

def NodeAddType(builder, type):
    builder.PrependInt32Slot(6, type, 0)

def AddType(builder, type):
    NodeAddType(builder, type)

def NodeAddExecutionProviderType(builder, executionProviderType):
    builder.PrependUOffsetTRelativeSlot(7, flatbuffers.number_types.UOffsetTFlags.py_type(executionProviderType), 0)

def AddExecutionProviderType(builder, executionProviderType):
    NodeAddExecutionProviderType(builder, executionProviderType)

def NodeAddInputs(builder, inputs):
    builder.PrependUOffsetTRelativeSlot(8, flatbuffers.number_types.UOffsetTFlags.py_type(inputs), 0)

def AddInputs(builder, inputs):
    NodeAddInputs(builder, inputs)

def NodeStartInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputsVector(builder, numElems: int) -> int:
    return NodeStartInputsVector(builder, numElems)

def NodeAddOutputs(builder, outputs):
    builder.PrependUOffsetTRelativeSlot(9, flatbuffers.number_types.UOffsetTFlags.py_type(outputs), 0)

def AddOutputs(builder, outputs):
    NodeAddOutputs(builder, outputs)

def NodeStartOutputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartOutputsVector(builder, numElems: int) -> int:
    return NodeStartOutputsVector(builder, numElems)

def NodeAddAttributes(builder, attributes):
    builder.PrependUOffsetTRelativeSlot(10, flatbuffers.number_types.UOffsetTFlags.py_type(attributes), 0)

def AddAttributes(builder, attributes):
    NodeAddAttributes(builder, attributes)

def NodeStartAttributesVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartAttributesVector(builder, numElems: int) -> int:
    return NodeStartAttributesVector(builder, numElems)

def NodeAddInputArgCounts(builder, inputArgCounts):
    builder.PrependUOffsetTRelativeSlot(11, flatbuffers.number_types.UOffsetTFlags.py_type(inputArgCounts), 0)

def AddInputArgCounts(builder, inputArgCounts):
    NodeAddInputArgCounts(builder, inputArgCounts)

def NodeStartInputArgCountsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartInputArgCountsVector(builder, numElems: int) -> int:
    return NodeStartInputArgCountsVector(builder, numElems)

def NodeAddImplicitInputs(builder, implicitInputs):
    builder.PrependUOffsetTRelativeSlot(12, flatbuffers.number_types.UOffsetTFlags.py_type(implicitInputs), 0)

def AddImplicitInputs(builder, implicitInputs):
    NodeAddImplicitInputs(builder, implicitInputs)

def NodeStartImplicitInputsVector(builder, numElems):
    return builder.StartVector(4, numElems, 4)

def StartImplicitInputsVector(builder, numElems: int) -> int:
    return NodeStartImplicitInputsVector(builder, numElems)

def NodeEnd(builder):
    return builder.EndObject()

def End(builder):
    return NodeEnd(builder)
