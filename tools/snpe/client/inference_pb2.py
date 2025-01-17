# Copyright (c) OpenMMLab. All rights reserved.
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: inference.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database

# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()

DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(
    b'\n\x0finference.proto\x12\x08mmdeploy"\x91\x01\n\x05Model\x12\x11\n\x04name\x18\x01 \x01(\tH\x00\x88\x01\x01\x12\x0f\n\x07weights\x18\x02 \x01(\x0c\x12+\n\x06\x64\x65vice\x18\x03 \x01(\x0e\x32\x16.mmdeploy.Model.DeviceH\x01\x88\x01\x01"#\n\x06\x44\x65vice\x12\x07\n\x03\x43PU\x10\x00\x12\x07\n\x03GPU\x10\x01\x12\x07\n\x03\x44SP\x10\x02\x42\x07\n\x05_nameB\t\n\x07_device"\x07\n\x05\x45mpty"Q\n\x06Tensor\x12\x0c\n\x04name\x18\x01 \x01(\t\x12\x12\n\x05\x64type\x18\x02 \x01(\tH\x00\x88\x01\x01\x12\x0c\n\x04\x64\x61ta\x18\x03 \x01(\x0c\x12\r\n\x05shape\x18\x04 \x03(\x05\x42\x08\n\x06_dtype",\n\nTensorList\x12\x1e\n\x04\x64\x61ta\x18\x01 \x03(\x0b\x32\x10.mmdeploy.Tensor"E\n\x05Reply\x12\x0e\n\x06status\x18\x01 \x01(\x05\x12\x0c\n\x04info\x18\x02 \x01(\t\x12\x1e\n\x04\x64\x61ta\x18\x03 \x03(\x0b\x32\x10.mmdeploy.Tensor"\x16\n\x05Names\x12\r\n\x05names\x18\x01 \x03(\t2\xfb\x01\n\tInference\x12*\n\x04\x45\x63ho\x12\x0f.mmdeploy.Empty\x1a\x0f.mmdeploy.Reply"\x00\x12*\n\x04Init\x12\x0f.mmdeploy.Model\x1a\x0f.mmdeploy.Reply"\x00\x12\x31\n\x0bOutputNames\x12\x0f.mmdeploy.Empty\x1a\x0f.mmdeploy.Names"\x00\x12\x34\n\tInference\x12\x14.mmdeploy.TensorList\x1a\x0f.mmdeploy.Reply"\x00\x12-\n\x07\x44\x65stroy\x12\x0f.mmdeploy.Empty\x1a\x0f.mmdeploy.Reply"\x00\x42%\n\rmmdeploy.snpeB\x0bSNPEWrapperP\x01\xa2\x02\x04SNPEb\x06proto3'
)

_MODEL = DESCRIPTOR.message_types_by_name["Model"]
_EMPTY = DESCRIPTOR.message_types_by_name["Empty"]
_TENSOR = DESCRIPTOR.message_types_by_name["Tensor"]
_TENSORLIST = DESCRIPTOR.message_types_by_name["TensorList"]
_REPLY = DESCRIPTOR.message_types_by_name["Reply"]
_NAMES = DESCRIPTOR.message_types_by_name["Names"]
_MODEL_DEVICE = _MODEL.enum_types_by_name["Device"]
Model = _reflection.GeneratedProtocolMessageType(
    "Model",
    (_message.Message,),
    {
        "DESCRIPTOR": _MODEL,
        "__module__": "inference_pb2"
        # @@protoc_insertion_point(class_scope:mmdeploy.Model)
    },
)
_sym_db.RegisterMessage(Model)

Empty = _reflection.GeneratedProtocolMessageType(
    "Empty",
    (_message.Message,),
    {
        "DESCRIPTOR": _EMPTY,
        "__module__": "inference_pb2"
        # @@protoc_insertion_point(class_scope:mmdeploy.Empty)
    },
)
_sym_db.RegisterMessage(Empty)

Tensor = _reflection.GeneratedProtocolMessageType(
    "Tensor",
    (_message.Message,),
    {
        "DESCRIPTOR": _TENSOR,
        "__module__": "inference_pb2"
        # @@protoc_insertion_point(class_scope:mmdeploy.Tensor)
    },
)
_sym_db.RegisterMessage(Tensor)

TensorList = _reflection.GeneratedProtocolMessageType(
    "TensorList",
    (_message.Message,),
    {
        "DESCRIPTOR": _TENSORLIST,
        "__module__": "inference_pb2"
        # @@protoc_insertion_point(class_scope:mmdeploy.TensorList)
    },
)
_sym_db.RegisterMessage(TensorList)

Reply = _reflection.GeneratedProtocolMessageType(
    "Reply",
    (_message.Message,),
    {
        "DESCRIPTOR": _REPLY,
        "__module__": "inference_pb2"
        # @@protoc_insertion_point(class_scope:mmdeploy.Reply)
    },
)
_sym_db.RegisterMessage(Reply)

Names = _reflection.GeneratedProtocolMessageType(
    "Names",
    (_message.Message,),
    {
        "DESCRIPTOR": _NAMES,
        "__module__": "inference_pb2"
        # @@protoc_insertion_point(class_scope:mmdeploy.Names)
    },
)
_sym_db.RegisterMessage(Names)

_INFERENCE = DESCRIPTOR.services_by_name["Inference"]
if _descriptor._USE_C_DESCRIPTORS == False:
    DESCRIPTOR._options = None
    DESCRIPTOR._serialized_options = b"\n\rmmdeploy.snpeB\013SNPEWrapperP\001\242\002\004SNPE"
    _MODEL._serialized_start = 30
    _MODEL._serialized_end = 175
    _MODEL_DEVICE._serialized_start = 120
    _MODEL_DEVICE._serialized_end = 155
    _EMPTY._serialized_start = 177
    _EMPTY._serialized_end = 184
    _TENSOR._serialized_start = 186
    _TENSOR._serialized_end = 267
    _TENSORLIST._serialized_start = 269
    _TENSORLIST._serialized_end = 313
    _REPLY._serialized_start = 315
    _REPLY._serialized_end = 384
    _NAMES._serialized_start = 386
    _NAMES._serialized_end = 408
    _INFERENCE._serialized_start = 411
    _INFERENCE._serialized_end = 662
# @@protoc_insertion_point(module_scope)
