я
ЭЃ
8
Const
output"dtype"
valuetensor"
dtypetype

NoOp
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype
О
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 

VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 "serve*2.3.12v2.3.0-54-gfcc4b966f18Чі
v
dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
ш*
shared_namedense/kernel
o
 dense/kernel/Read/ReadVariableOpReadVariableOpdense/kernel* 
_output_shapes
:
ш*
dtype0
m

dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_name
dense/bias
f
dense/bias/Read/ReadVariableOpReadVariableOp
dense/bias*
_output_shapes	
:*
dtype0
z
dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:
*
shared_namedense_1/kernel
s
"dense_1/kernel/Read/ReadVariableOpReadVariableOpdense_1/kernel* 
_output_shapes
:
*
dtype0
q
dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_namedense_1/bias
j
 dense_1/bias/Read/ReadVariableOpReadVariableOpdense_1/bias*
_output_shapes	
:*
dtype0

NoOpNoOp

ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*е
valueЫBШ BС
Н
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
 
h

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
h

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api

	0

1
2
3
 

	0

1
2
3
­
layer_metrics
non_trainable_variables
layer_regularization_losses
	variables
metrics
regularization_losses

layers
trainable_variables
 
XV
VARIABLE_VALUEdense/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
dense/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE

	0

1
 

	0

1
­
layer_metrics
non_trainable_variables
layer_regularization_losses
metrics
	variables
regularization_losses

layers
trainable_variables
ZX
VARIABLE_VALUEdense_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEdense_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
­
layer_metrics
 non_trainable_variables
!layer_regularization_losses
"metrics
	variables
regularization_losses

#layers
trainable_variables
 
 
 
 

0
1
2
 
 
 
 
 
 
 
 
 
 
|
serving_default_input_1Placeholder*(
_output_shapes
:џџџџџџџџџш*
dtype0*
shape:џџџџџџџџџш
ѓ
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1dense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *-
f(R&
$__inference_signature_wrapper_544982
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
Ї
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename dense/kernel/Read/ReadVariableOpdense/bias/Read/ReadVariableOp"dense_1/kernel/Read/ReadVariableOp dense_1/bias/Read/ReadVariableOpConst*
Tin

2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *(
f#R!
__inference__traced_save_545229
в
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenamedense/kernel
dense/biasdense_1/kerneldense_1/bias*
Tin	
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *+
f&R$
"__inference__traced_restore_545251Ни
ЇK
Е
H__inference_functional_3_layer_call_and_return_conditional_losses_545102

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2Ё
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/BiasAddh
	dense/EluEludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
	dense/Elu
dense/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense/ActivityRegularizer/Const
dense/ActivityRegularizer/AbsAbsdense/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/ActivityRegularizer/Abs
!dense/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!dense/ActivityRegularizer/Const_1Е
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0*dense/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/Sum
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72!
dense/ActivityRegularizer/mul/xИ
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/mulЕ
dense/ActivityRegularizer/addAddV2(dense/ActivityRegularizer/Const:output:0!dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/add
 dense/ActivityRegularizer/SquareSquaredense/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 dense/ActivityRegularizer/Square
!dense/ActivityRegularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2#
!dense/ActivityRegularizer/Const_2М
dense/ActivityRegularizer/Sum_1Sum$dense/ActivityRegularizer/Square:y:0*dense/ActivityRegularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/Sum_1
!dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72#
!dense/ActivityRegularizer/mul_1/xР
dense/ActivityRegularizer/mul_1Mul*dense/ActivityRegularizer/mul_1/x:output:0(dense/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/mul_1Д
dense/ActivityRegularizer/add_1AddV2!dense/ActivityRegularizer/add:z:0#dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/add_1
dense/ActivityRegularizer/ShapeShapedense/Elu:activations:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/ShapeЈ
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackЌ
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1Ќ
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2ў
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceЊ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/CastЛ
!dense/ActivityRegularizer/truedivRealDiv#dense/ActivityRegularizer/add_1:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЇ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpЂ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddn
dense_1/EluEludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/Elu
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_1/ActivityRegularizer/Const
dense_1/ActivityRegularizer/AbsAbsdense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
dense_1/ActivityRegularizer/Abs
#dense_1/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_1/ActivityRegularizer/Const_1Н
dense_1/ActivityRegularizer/SumSum#dense_1/ActivityRegularizer/Abs:y:0,dense_1/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/Sum
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72#
!dense_1/ActivityRegularizer/mul/xР
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/mulН
dense_1/ActivityRegularizer/addAddV2*dense_1/ActivityRegularizer/Const:output:0#dense_1/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/add 
"dense_1/ActivityRegularizer/SquareSquaredense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2$
"dense_1/ActivityRegularizer/Square
#dense_1/ActivityRegularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_1/ActivityRegularizer/Const_2Ф
!dense_1/ActivityRegularizer/Sum_1Sum&dense_1/ActivityRegularizer/Square:y:0,dense_1/ActivityRegularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_1/ActivityRegularizer/Sum_1
#dense_1/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72%
#dense_1/ActivityRegularizer/mul_1/xШ
!dense_1/ActivityRegularizer/mul_1Mul,dense_1/ActivityRegularizer/mul_1/x:output:0*dense_1/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_1/ActivityRegularizer/mul_1М
!dense_1/ActivityRegularizer/add_1AddV2#dense_1/ActivityRegularizer/add:z:0%dense_1/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_1/ActivityRegularizer/add_1
!dense_1/ActivityRegularizer/ShapeShapedense_1/Elu:activations:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/ShapeЌ
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stackА
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1А
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_sliceА
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/CastУ
#dense_1/ActivityRegularizer/truedivRealDiv%dense_1/ActivityRegularizer/add_1:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivn
IdentityIdentitydense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityl

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_1n

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:џџџџџџџџџш:::::P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs

G
-__inference_dense_activity_regularizer_544715
self
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const:
AbsAbsself*
T0*
_output_shapes
:2
Abs>
RankRankAbs:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2
rangeK
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulM
addAddV2Const:output:0mul:z:0*
T0*
_output_shapes
: 2
addC
SquareSquareself*
T0*
_output_shapes
:2
SquareE
Rank_1Rank
Square:y:0*
T0*
_output_shapes
: 2
Rank_1`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2	
range_1T
Sum_1Sum
Square:y:0range_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72	
mul_1/xX
mul_1Mulmul_1/x:output:0Sum_1:output:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2add:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
Т,

H__inference_functional_3_layer_call_and_return_conditional_losses_544907

inputs
dense_544878
dense_544880
dense_1_544891
dense_1_544893
identity

identity_1

identity_2Ђdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_544878dense_544880*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5447542
dense/StatefulPartitionedCallю
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *6
f1R/
-__inference_dense_activity_regularizer_5447152+
)dense/ActivityRegularizer/PartitionedCall
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/ShapeЈ
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackЌ
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1Ќ
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2ў
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceЊ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/CastЪ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivА
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_544891dense_1_544893*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5448012!
dense_1/StatefulPartitionedCallі
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_1_activity_regularizer_5447392-
+dense_1/ActivityRegularizer/PartitionedCall
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/ShapeЌ
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stackА
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1А
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_sliceА
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castв
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivП
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЎ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1А

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:џџџџџџџџџш::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
к
{
&__inference_dense_layer_call_fn_545152

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5447542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџш::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
Џ
 
-__inference_functional_3_layer_call_fn_545117

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_5449072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџш::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
­
Ћ
C__inference_dense_1_layer_call_and_return_conditional_losses_544801

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
В
Ё
-__inference_functional_3_layer_call_fn_544920
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_5449072
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџш::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџш
!
_user_specified_name	input_1
о
}
(__inference_dense_1_layer_call_fn_545183

inputs
unknown
	unknown_0
identityЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5448012
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Е
Ќ
"__inference__traced_restore_545251
file_prefix!
assignvariableop_dense_kernel!
assignvariableop_1_dense_bias%
!assignvariableop_2_dense_1_kernel#
assignvariableop_3_dense_1_bias

identity_5ЂAssignVariableOpЂAssignVariableOp_1ЂAssignVariableOp_2ЂAssignVariableOp_3
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
RestoreV2/shape_and_slicesФ
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*(
_output_shapes
:::::*
dtypes	
22
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity
AssignVariableOpAssignVariableOpassignvariableop_dense_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1Ђ
AssignVariableOp_1AssignVariableOpassignvariableop_1_dense_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2І
AssignVariableOp_2AssignVariableOp!assignvariableop_2_dense_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3Є
AssignVariableOp_3AssignVariableOpassignvariableop_3_dense_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOpК

Identity_4Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_4Ќ

Identity_5IdentityIdentity_4:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3*
T0*
_output_shapes
: 2

Identity_5"!

identity_5Identity_5:output:0*%
_input_shapes
: ::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_3:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
ќ

$__inference_signature_wrapper_544982
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCallэ
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 **
f%R#
!__inference__wrapped_model_5446912
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџш::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџш
!
_user_specified_name	input_1
В
Ё
-__inference_functional_3_layer_call_fn_544967
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_5449542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџш::::22
StatefulPartitionedCallStatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџш
!
_user_specified_name	input_1
­
Ћ
C__inference_dense_1_layer_call_and_return_conditional_losses_545174

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџ:::P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
Џ
 
-__inference_functional_3_layer_call_fn_545132

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
identityЂStatefulPartitionedCall
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2*
Tin	
2*
Tout
2*
_collective_manager_ids
 *,
_output_shapes
:џџџџџџџџџ: : *&
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *Q
fLRJ
H__inference_functional_3_layer_call_and_return_conditional_losses_5449542
StatefulPartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџш::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
ЇK
Е
H__inference_functional_3_layer_call_and_return_conditional_losses_545042

inputs(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2Ё
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
dense/MatMul/ReadVariableOp
dense/MatMulMatMulinputs#dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/MatMul
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
dense/BiasAdd/ReadVariableOp
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/BiasAddh
	dense/EluEludense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
	dense/Elu
dense/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2!
dense/ActivityRegularizer/Const
dense/ActivityRegularizer/AbsAbsdense/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense/ActivityRegularizer/Abs
!dense/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2#
!dense/ActivityRegularizer/Const_1Е
dense/ActivityRegularizer/SumSum!dense/ActivityRegularizer/Abs:y:0*dense/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/Sum
dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72!
dense/ActivityRegularizer/mul/xИ
dense/ActivityRegularizer/mulMul(dense/ActivityRegularizer/mul/x:output:0&dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/mulЕ
dense/ActivityRegularizer/addAddV2(dense/ActivityRegularizer/Const:output:0!dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2
dense/ActivityRegularizer/add
 dense/ActivityRegularizer/SquareSquaredense/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2"
 dense/ActivityRegularizer/Square
!dense/ActivityRegularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2#
!dense/ActivityRegularizer/Const_2М
dense/ActivityRegularizer/Sum_1Sum$dense/ActivityRegularizer/Square:y:0*dense/ActivityRegularizer/Const_2:output:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/Sum_1
!dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72#
!dense/ActivityRegularizer/mul_1/xР
dense/ActivityRegularizer/mul_1Mul*dense/ActivityRegularizer/mul_1/x:output:0(dense/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/mul_1Д
dense/ActivityRegularizer/add_1AddV2!dense/ActivityRegularizer/add:z:0#dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2!
dense/ActivityRegularizer/add_1
dense/ActivityRegularizer/ShapeShapedense/Elu:activations:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/ShapeЈ
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackЌ
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1Ќ
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2ў
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceЊ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/CastЛ
!dense/ActivityRegularizer/truedivRealDiv#dense/ActivityRegularizer/add_1:z:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivЇ
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02
dense_1/MatMul/ReadVariableOp
dense_1/MatMulMatMuldense/Elu:activations:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/MatMulЅ
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02 
dense_1/BiasAdd/ReadVariableOpЂ
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/BiasAddn
dense_1/EluEludense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
dense_1/Elu
!dense_1/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2#
!dense_1/ActivityRegularizer/Const
dense_1/ActivityRegularizer/AbsAbsdense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2!
dense_1/ActivityRegularizer/Abs
#dense_1/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_1/ActivityRegularizer/Const_1Н
dense_1/ActivityRegularizer/SumSum#dense_1/ActivityRegularizer/Abs:y:0,dense_1/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/Sum
!dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72#
!dense_1/ActivityRegularizer/mul/xР
dense_1/ActivityRegularizer/mulMul*dense_1/ActivityRegularizer/mul/x:output:0(dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/mulН
dense_1/ActivityRegularizer/addAddV2*dense_1/ActivityRegularizer/Const:output:0#dense_1/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2!
dense_1/ActivityRegularizer/add 
"dense_1/ActivityRegularizer/SquareSquaredense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2$
"dense_1/ActivityRegularizer/Square
#dense_1/ActivityRegularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       2%
#dense_1/ActivityRegularizer/Const_2Ф
!dense_1/ActivityRegularizer/Sum_1Sum&dense_1/ActivityRegularizer/Square:y:0,dense_1/ActivityRegularizer/Const_2:output:0*
T0*
_output_shapes
: 2#
!dense_1/ActivityRegularizer/Sum_1
#dense_1/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72%
#dense_1/ActivityRegularizer/mul_1/xШ
!dense_1/ActivityRegularizer/mul_1Mul,dense_1/ActivityRegularizer/mul_1/x:output:0*dense_1/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: 2#
!dense_1/ActivityRegularizer/mul_1М
!dense_1/ActivityRegularizer/add_1AddV2#dense_1/ActivityRegularizer/add:z:0%dense_1/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2#
!dense_1/ActivityRegularizer/add_1
!dense_1/ActivityRegularizer/ShapeShapedense_1/Elu:activations:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/ShapeЌ
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stackА
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1А
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_sliceА
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/CastУ
#dense_1/ActivityRegularizer/truedivRealDiv%dense_1/ActivityRegularizer/add_1:z:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivn
IdentityIdentitydense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityl

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_1n

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:џџџџџџџџџш:::::P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
ћX
Ѓ
!__inference__wrapped_model_544691
input_15
1functional_3_dense_matmul_readvariableop_resource6
2functional_3_dense_biasadd_readvariableop_resource7
3functional_3_dense_1_matmul_readvariableop_resource8
4functional_3_dense_1_biasadd_readvariableop_resource
identityШ
(functional_3/dense/MatMul/ReadVariableOpReadVariableOp1functional_3_dense_matmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02*
(functional_3/dense/MatMul/ReadVariableOpЎ
functional_3/dense/MatMulMatMulinput_10functional_3/dense/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_3/dense/MatMulЦ
)functional_3/dense/BiasAdd/ReadVariableOpReadVariableOp2functional_3_dense_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02+
)functional_3/dense/BiasAdd/ReadVariableOpЮ
functional_3/dense/BiasAddBiasAdd#functional_3/dense/MatMul:product:01functional_3/dense/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_3/dense/BiasAdd
functional_3/dense/EluElu#functional_3/dense/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_3/dense/EluЁ
,functional_3/dense/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2.
,functional_3/dense/ActivityRegularizer/ConstИ
*functional_3/dense/ActivityRegularizer/AbsAbs$functional_3/dense/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2,
*functional_3/dense/ActivityRegularizer/AbsБ
.functional_3/dense/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       20
.functional_3/dense/ActivityRegularizer/Const_1щ
*functional_3/dense/ActivityRegularizer/SumSum.functional_3/dense/ActivityRegularizer/Abs:y:07functional_3/dense/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: 2,
*functional_3/dense/ActivityRegularizer/SumЁ
,functional_3/dense/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72.
,functional_3/dense/ActivityRegularizer/mul/xь
*functional_3/dense/ActivityRegularizer/mulMul5functional_3/dense/ActivityRegularizer/mul/x:output:03functional_3/dense/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2,
*functional_3/dense/ActivityRegularizer/mulщ
*functional_3/dense/ActivityRegularizer/addAddV25functional_3/dense/ActivityRegularizer/Const:output:0.functional_3/dense/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2,
*functional_3/dense/ActivityRegularizer/addС
-functional_3/dense/ActivityRegularizer/SquareSquare$functional_3/dense/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2/
-functional_3/dense/ActivityRegularizer/SquareБ
.functional_3/dense/ActivityRegularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       20
.functional_3/dense/ActivityRegularizer/Const_2№
,functional_3/dense/ActivityRegularizer/Sum_1Sum1functional_3/dense/ActivityRegularizer/Square:y:07functional_3/dense/ActivityRegularizer/Const_2:output:0*
T0*
_output_shapes
: 2.
,functional_3/dense/ActivityRegularizer/Sum_1Ѕ
.functional_3/dense/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'720
.functional_3/dense/ActivityRegularizer/mul_1/xє
,functional_3/dense/ActivityRegularizer/mul_1Mul7functional_3/dense/ActivityRegularizer/mul_1/x:output:05functional_3/dense/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: 2.
,functional_3/dense/ActivityRegularizer/mul_1ш
,functional_3/dense/ActivityRegularizer/add_1AddV2.functional_3/dense/ActivityRegularizer/add:z:00functional_3/dense/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 2.
,functional_3/dense/ActivityRegularizer/add_1А
,functional_3/dense/ActivityRegularizer/ShapeShape$functional_3/dense/Elu:activations:0*
T0*
_output_shapes
:2.
,functional_3/dense/ActivityRegularizer/ShapeТ
:functional_3/dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2<
:functional_3/dense/ActivityRegularizer/strided_slice/stackЦ
<functional_3/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2>
<functional_3/dense/ActivityRegularizer/strided_slice/stack_1Ц
<functional_3/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2>
<functional_3/dense/ActivityRegularizer/strided_slice/stack_2Ь
4functional_3/dense/ActivityRegularizer/strided_sliceStridedSlice5functional_3/dense/ActivityRegularizer/Shape:output:0Cfunctional_3/dense/ActivityRegularizer/strided_slice/stack:output:0Efunctional_3/dense/ActivityRegularizer/strided_slice/stack_1:output:0Efunctional_3/dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask26
4functional_3/dense/ActivityRegularizer/strided_sliceб
+functional_3/dense/ActivityRegularizer/CastCast=functional_3/dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2-
+functional_3/dense/ActivityRegularizer/Castя
.functional_3/dense/ActivityRegularizer/truedivRealDiv0functional_3/dense/ActivityRegularizer/add_1:z:0/functional_3/dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 20
.functional_3/dense/ActivityRegularizer/truedivЮ
*functional_3/dense_1/MatMul/ReadVariableOpReadVariableOp3functional_3_dense_1_matmul_readvariableop_resource* 
_output_shapes
:
*
dtype02,
*functional_3/dense_1/MatMul/ReadVariableOpб
functional_3/dense_1/MatMulMatMul$functional_3/dense/Elu:activations:02functional_3/dense_1/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_3/dense_1/MatMulЬ
+functional_3/dense_1/BiasAdd/ReadVariableOpReadVariableOp4functional_3_dense_1_biasadd_readvariableop_resource*
_output_shapes	
:*
dtype02-
+functional_3/dense_1/BiasAdd/ReadVariableOpж
functional_3/dense_1/BiasAddBiasAdd%functional_3/dense_1/MatMul:product:03functional_3/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_3/dense_1/BiasAdd
functional_3/dense_1/EluElu%functional_3/dense_1/BiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
functional_3/dense_1/EluЅ
.functional_3/dense_1/ActivityRegularizer/ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    20
.functional_3/dense_1/ActivityRegularizer/ConstО
,functional_3/dense_1/ActivityRegularizer/AbsAbs&functional_3/dense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2.
,functional_3/dense_1/ActivityRegularizer/AbsЕ
0functional_3/dense_1/ActivityRegularizer/Const_1Const*
_output_shapes
:*
dtype0*
valueB"       22
0functional_3/dense_1/ActivityRegularizer/Const_1ё
,functional_3/dense_1/ActivityRegularizer/SumSum0functional_3/dense_1/ActivityRegularizer/Abs:y:09functional_3/dense_1/ActivityRegularizer/Const_1:output:0*
T0*
_output_shapes
: 2.
,functional_3/dense_1/ActivityRegularizer/SumЅ
.functional_3/dense_1/ActivityRegularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'720
.functional_3/dense_1/ActivityRegularizer/mul/xє
,functional_3/dense_1/ActivityRegularizer/mulMul7functional_3/dense_1/ActivityRegularizer/mul/x:output:05functional_3/dense_1/ActivityRegularizer/Sum:output:0*
T0*
_output_shapes
: 2.
,functional_3/dense_1/ActivityRegularizer/mulё
,functional_3/dense_1/ActivityRegularizer/addAddV27functional_3/dense_1/ActivityRegularizer/Const:output:00functional_3/dense_1/ActivityRegularizer/mul:z:0*
T0*
_output_shapes
: 2.
,functional_3/dense_1/ActivityRegularizer/addЧ
/functional_3/dense_1/ActivityRegularizer/SquareSquare&functional_3/dense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ21
/functional_3/dense_1/ActivityRegularizer/SquareЕ
0functional_3/dense_1/ActivityRegularizer/Const_2Const*
_output_shapes
:*
dtype0*
valueB"       22
0functional_3/dense_1/ActivityRegularizer/Const_2ј
.functional_3/dense_1/ActivityRegularizer/Sum_1Sum3functional_3/dense_1/ActivityRegularizer/Square:y:09functional_3/dense_1/ActivityRegularizer/Const_2:output:0*
T0*
_output_shapes
: 20
.functional_3/dense_1/ActivityRegularizer/Sum_1Љ
0functional_3/dense_1/ActivityRegularizer/mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'722
0functional_3/dense_1/ActivityRegularizer/mul_1/xќ
.functional_3/dense_1/ActivityRegularizer/mul_1Mul9functional_3/dense_1/ActivityRegularizer/mul_1/x:output:07functional_3/dense_1/ActivityRegularizer/Sum_1:output:0*
T0*
_output_shapes
: 20
.functional_3/dense_1/ActivityRegularizer/mul_1№
.functional_3/dense_1/ActivityRegularizer/add_1AddV20functional_3/dense_1/ActivityRegularizer/add:z:02functional_3/dense_1/ActivityRegularizer/mul_1:z:0*
T0*
_output_shapes
: 20
.functional_3/dense_1/ActivityRegularizer/add_1Ж
.functional_3/dense_1/ActivityRegularizer/ShapeShape&functional_3/dense_1/Elu:activations:0*
T0*
_output_shapes
:20
.functional_3/dense_1/ActivityRegularizer/ShapeЦ
<functional_3/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2>
<functional_3/dense_1/ActivityRegularizer/strided_slice/stackЪ
>functional_3/dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2@
>functional_3/dense_1/ActivityRegularizer/strided_slice/stack_1Ъ
>functional_3/dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2@
>functional_3/dense_1/ActivityRegularizer/strided_slice/stack_2и
6functional_3/dense_1/ActivityRegularizer/strided_sliceStridedSlice7functional_3/dense_1/ActivityRegularizer/Shape:output:0Efunctional_3/dense_1/ActivityRegularizer/strided_slice/stack:output:0Gfunctional_3/dense_1/ActivityRegularizer/strided_slice/stack_1:output:0Gfunctional_3/dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask28
6functional_3/dense_1/ActivityRegularizer/strided_sliceз
-functional_3/dense_1/ActivityRegularizer/CastCast?functional_3/dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2/
-functional_3/dense_1/ActivityRegularizer/Castї
0functional_3/dense_1/ActivityRegularizer/truedivRealDiv2functional_3/dense_1/ActivityRegularizer/add_1:z:01functional_3/dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 22
0functional_3/dense_1/ActivityRegularizer/truediv{
IdentityIdentity&functional_3/dense_1/Elu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*7
_input_shapes&
$:џџџџџџџџџш:::::Q M
(
_output_shapes
:џџџџџџџџџш
!
_user_specified_name	input_1
Х,

H__inference_functional_3_layer_call_and_return_conditional_losses_544872
input_1
dense_544843
dense_544845
dense_1_544856
dense_1_544858
identity

identity_1

identity_2Ђdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_544843dense_544845*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5447542
dense/StatefulPartitionedCallю
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *6
f1R/
-__inference_dense_activity_regularizer_5447152+
)dense/ActivityRegularizer/PartitionedCall
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/ShapeЈ
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackЌ
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1Ќ
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2ў
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceЊ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/CastЪ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivА
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_544856dense_1_544858*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5448012!
dense_1/StatefulPartitionedCallі
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_1_activity_regularizer_5447392-
+dense_1/ActivityRegularizer/PartitionedCall
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/ShapeЌ
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stackА
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1А
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_sliceА
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castв
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivП
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЎ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1А

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:џџџџџџџџџш::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџш
!
_user_specified_name	input_1
Ћ
Љ
A__inference_dense_layer_call_and_return_conditional_losses_545143

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџш:::P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
х

Ќ
G__inference_dense_1_layer_call_and_return_all_conditional_losses_545194

inputs
unknown
	unknown_0
identity

identity_1ЂStatefulPartitionedCallє
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5448012
StatefulPartitionedCallЖ
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_1_activity_regularizer_5447392
PartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*/
_input_shapes
:џџџџџџџџџ::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџ
 
_user_specified_nameinputs
ї
 
__inference__traced_save_545229
file_prefix+
'savev2_dense_kernel_read_readvariableop)
%savev2_dense_bias_read_readvariableop-
)savev2_dense_1_kernel_read_readvariableop+
'savev2_dense_1_bias_read_readvariableop
savev2_const

identity_1ЂMergeV2Checkpoints
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Const
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*<
value3B1 B+_temp_4cad25f2a96f4df388d0acf31d35d3e1/part2	
Const_1
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shardІ
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename§
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:*
dtype0*
valueBB B B B B 2
SaveV2/shape_and_slicesт
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_dense_kernel_read_readvariableop%savev2_dense_bias_read_readvariableop)savev2_dense_1_kernel_read_readvariableop'savev2_dense_1_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes	
22
SaveV2К
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixesЁ
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*=
_input_shapes,
*: :
ш::
:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:&"
 
_output_shapes
:
ш:!

_output_shapes	
::&"
 
_output_shapes
:
:!

_output_shapes	
::

_output_shapes
: 
Т,

H__inference_functional_3_layer_call_and_return_conditional_losses_544954

inputs
dense_544925
dense_544927
dense_1_544938
dense_1_544940
identity

identity_1

identity_2Ђdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinputsdense_544925dense_544927*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5447542
dense/StatefulPartitionedCallю
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *6
f1R/
-__inference_dense_activity_regularizer_5447152+
)dense/ActivityRegularizer/PartitionedCall
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/ShapeЈ
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackЌ
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1Ќ
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2ў
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceЊ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/CastЪ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivА
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_544938dense_1_544940*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5448012!
dense_1/StatefulPartitionedCallі
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_1_activity_regularizer_5447392-
+dense_1/ActivityRegularizer/PartitionedCall
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/ShapeЌ
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stackА
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1А
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_sliceА
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castв
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivП
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЎ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1А

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:џџџџџџџџџш::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs
Ћ
Љ
A__inference_dense_layer_call_and_return_conditional_losses_544754

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource* 
_output_shapes
:
ш*
dtype02
MatMul/ReadVariableOpt
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2
MatMul
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes	
:*
dtype02
BiasAdd/ReadVariableOp
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:џџџџџџџџџ2	
BiasAddV
EluEluBiasAdd:output:0*
T0*(
_output_shapes
:џџџџџџџџџ2
Eluf
IdentityIdentityElu:activations:0*
T0*(
_output_shapes
:џџџџџџџџџ2

Identity"
identityIdentity:output:0*/
_input_shapes
:џџџџџџџџџш:::P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs

I
/__inference_dense_1_activity_regularizer_544739
self
identityS
ConstConst*
_output_shapes
: *
dtype0*
valueB
 *    2
Const:
AbsAbsself*
T0*
_output_shapes
:2
Abs>
RankRankAbs:y:0*
T0*
_output_shapes
: 2
Rank\
range/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range/start\
range/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range/deltaw
rangeRangerange/start:output:0Rank:output:0range/delta:output:0*#
_output_shapes
:џџџџџџџџџ2
rangeK
SumSumAbs:y:0range:output:0*
T0*
_output_shapes
: 2
SumS
mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72
mul/xP
mulMulmul/x:output:0Sum:output:0*
T0*
_output_shapes
: 2
mulM
addAddV2Const:output:0mul:z:0*
T0*
_output_shapes
: 2
addC
SquareSquareself*
T0*
_output_shapes
:2
SquareE
Rank_1Rank
Square:y:0*
T0*
_output_shapes
: 2
Rank_1`
range_1/startConst*
_output_shapes
: *
dtype0*
value	B : 2
range_1/start`
range_1/deltaConst*
_output_shapes
: *
dtype0*
value	B :2
range_1/delta
range_1Rangerange_1/start:output:0Rank_1:output:0range_1/delta:output:0*#
_output_shapes
:џџџџџџџџџ2	
range_1T
Sum_1Sum
Square:y:0range_1:output:0*
T0*
_output_shapes
: 2
Sum_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *ЌХ'72	
mul_1/xX
mul_1Mulmul_1/x:output:0Sum_1:output:0*
T0*
_output_shapes
: 2
mul_1L
add_1AddV2add:z:0	mul_1:z:0*
T0*
_output_shapes
: 2
add_1L
IdentityIdentity	add_1:z:0*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
::> :

_output_shapes
:

_user_specified_nameself
Х,

H__inference_functional_3_layer_call_and_return_conditional_losses_544840
input_1
dense_544777
dense_544779
dense_1_544824
dense_1_544826
identity

identity_1

identity_2Ђdense/StatefulPartitionedCallЂdense_1/StatefulPartitionedCall
dense/StatefulPartitionedCallStatefulPartitionedCallinput_1dense_544777dense_544779*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5447542
dense/StatefulPartitionedCallю
)dense/ActivityRegularizer/PartitionedCallPartitionedCall&dense/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *6
f1R/
-__inference_dense_activity_regularizer_5447152+
)dense/ActivityRegularizer/PartitionedCall
dense/ActivityRegularizer/ShapeShape&dense/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2!
dense/ActivityRegularizer/ShapeЈ
-dense/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2/
-dense/ActivityRegularizer/strided_slice/stackЌ
/dense/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_1Ќ
/dense/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:21
/dense/ActivityRegularizer/strided_slice/stack_2ў
'dense/ActivityRegularizer/strided_sliceStridedSlice(dense/ActivityRegularizer/Shape:output:06dense/ActivityRegularizer/strided_slice/stack:output:08dense/ActivityRegularizer/strided_slice/stack_1:output:08dense/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2)
'dense/ActivityRegularizer/strided_sliceЊ
dense/ActivityRegularizer/CastCast0dense/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2 
dense/ActivityRegularizer/CastЪ
!dense/ActivityRegularizer/truedivRealDiv2dense/ActivityRegularizer/PartitionedCall:output:0"dense/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2#
!dense/ActivityRegularizer/truedivА
dense_1/StatefulPartitionedCallStatefulPartitionedCall&dense/StatefulPartitionedCall:output:0dense_1_544824dense_1_544826*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *L
fGRE
C__inference_dense_1_layer_call_and_return_conditional_losses_5448012!
dense_1/StatefulPartitionedCallі
+dense_1/ActivityRegularizer/PartitionedCallPartitionedCall(dense_1/StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *8
f3R1
/__inference_dense_1_activity_regularizer_5447392-
+dense_1/ActivityRegularizer/PartitionedCall
!dense_1/ActivityRegularizer/ShapeShape(dense_1/StatefulPartitionedCall:output:0*
T0*
_output_shapes
:2#
!dense_1/ActivityRegularizer/ShapeЌ
/dense_1/ActivityRegularizer/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 21
/dense_1/ActivityRegularizer/strided_slice/stackА
1dense_1/ActivityRegularizer/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_1А
1dense_1/ActivityRegularizer/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:23
1dense_1/ActivityRegularizer/strided_slice/stack_2
)dense_1/ActivityRegularizer/strided_sliceStridedSlice*dense_1/ActivityRegularizer/Shape:output:08dense_1/ActivityRegularizer/strided_slice/stack:output:0:dense_1/ActivityRegularizer/strided_slice/stack_1:output:0:dense_1/ActivityRegularizer/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2+
)dense_1/ActivityRegularizer/strided_sliceА
 dense_1/ActivityRegularizer/CastCast2dense_1/ActivityRegularizer/strided_slice:output:0*

DstT0*

SrcT0*
_output_shapes
: 2"
 dense_1/ActivityRegularizer/Castв
#dense_1/ActivityRegularizer/truedivRealDiv4dense_1/ActivityRegularizer/PartitionedCall:output:0$dense_1/ActivityRegularizer/Cast:y:0*
T0*
_output_shapes
: 2%
#dense_1/ActivityRegularizer/truedivП
IdentityIdentity(dense_1/StatefulPartitionedCall:output:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

IdentityЎ

Identity_1Identity%dense/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1А

Identity_2Identity'dense_1/ActivityRegularizer/truediv:z:0^dense/StatefulPartitionedCall ^dense_1/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*7
_input_shapes&
$:џџџџџџџџџш::::2>
dense/StatefulPartitionedCalldense/StatefulPartitionedCall2B
dense_1/StatefulPartitionedCalldense_1/StatefulPartitionedCall:Q M
(
_output_shapes
:џџџџџџџџџш
!
_user_specified_name	input_1
п

Њ
E__inference_dense_layer_call_and_return_all_conditional_losses_545163

inputs
unknown
	unknown_0
identity

identity_1ЂStatefulPartitionedCallђ
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *(
_output_shapes
:џџџџџџџџџ*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8 *J
fERC
A__inference_dense_layer_call_and_return_conditional_losses_5447542
StatefulPartitionedCallД
PartitionedCallPartitionedCall StatefulPartitionedCall:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8 *6
f1R/
-__inference_dense_activity_regularizer_5447152
PartitionedCall
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*(
_output_shapes
:џџџџџџџџџ2

Identityy

Identity_1IdentityPartitionedCall:output:0^StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*/
_input_shapes
:џџџџџџџџџш::22
StatefulPartitionedCallStatefulPartitionedCall:P L
(
_output_shapes
:џџџџџџџџџш
 
_user_specified_nameinputs"ИL
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*Ќ
serving_default
<
input_11
serving_default_input_1:0џџџџџџџџџш<
dense_11
StatefulPartitionedCall:0џџџџџџџџџtensorflow/serving/predict:q
о
layer-0
layer_with_weights-0
layer-1
layer_with_weights-1
layer-2
	variables
regularization_losses
trainable_variables
	keras_api

signatures
$_default_save_signature
*%&call_and_return_all_conditional_losses
&__call__"Ч
_tf_keras_networkЋ{"class_name": "Functional", "name": "functional_3", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_1", 0, 0]]}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Functional", "config": {"name": "functional_3", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}, "name": "input_1", "inbound_nodes": []}, {"class_name": "Dense", "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense", "inbound_nodes": [[["input_1", 0, 0, {}]]]}, {"class_name": "Dense", "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "kernel_constraint": null, "bias_constraint": null}, "name": "dense_1", "inbound_nodes": [[["dense", 0, 0, {}]]]}], "input_layers": [["input_1", 0, 0]], "output_layers": [["dense_1", 0, 0]]}}}
я"ь
_tf_keras_input_layerЬ{"class_name": "InputLayer", "name": "input_1", "dtype": "float32", "sparse": false, "ragged": false, "batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 1000]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}
О	

	kernel

bias
	variables
regularization_losses
trainable_variables
	keras_api
*'&call_and_return_all_conditional_losses
(__call__"
_tf_keras_layerџ{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 256, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 1000}}}, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 1000]}}
Р	

kernel
bias
	variables
regularization_losses
trainable_variables
	keras_api
*)&call_and_return_all_conditional_losses
*__call__"
_tf_keras_layer{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 128, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 256}}}, "activity_regularizer": {"class_name": "L1L2", "config": {"l1": 9.999999747378752e-06, "l2": 9.999999747378752e-06}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 256]}}
<
	0

1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
<
	0

1
2
3"
trackable_list_wrapper
Ъ
layer_metrics
non_trainable_variables
layer_regularization_losses
	variables
metrics
regularization_losses

layers
trainable_variables
&__call__
$_default_save_signature
*%&call_and_return_all_conditional_losses
&%"call_and_return_conditional_losses"
_generic_user_object
,
+serving_default"
signature_map
 :
ш2dense/kernel
:2
dense/bias
.
	0

1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
	0

1"
trackable_list_wrapper
Ъ
layer_metrics
non_trainable_variables
layer_regularization_losses
metrics
	variables
regularization_losses

layers
trainable_variables
(__call__
,activity_regularizer_fn
*'&call_and_return_all_conditional_losses
&-"call_and_return_conditional_losses"
_generic_user_object
": 
2dense_1/kernel
:2dense_1/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
Ъ
layer_metrics
 non_trainable_variables
!layer_regularization_losses
"metrics
	variables
regularization_losses

#layers
trainable_variables
*__call__
.activity_regularizer_fn
*)&call_and_return_all_conditional_losses
&/"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
5
0
1
2"
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
р2н
!__inference__wrapped_model_544691З
В
FullArgSpec
args 
varargsjargs
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *'Ђ$
"
input_1џџџџџџџџџш
ю2ы
H__inference_functional_3_layer_call_and_return_conditional_losses_544872
H__inference_functional_3_layer_call_and_return_conditional_losses_545042
H__inference_functional_3_layer_call_and_return_conditional_losses_544840
H__inference_functional_3_layer_call_and_return_conditional_losses_545102Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
2џ
-__inference_functional_3_layer_call_fn_544920
-__inference_functional_3_layer_call_fn_544967
-__inference_functional_3_layer_call_fn_545117
-__inference_functional_3_layer_call_fn_545132Р
ЗВГ
FullArgSpec1
args)&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults
p 

 

kwonlyargs 
kwonlydefaultsЊ 
annotationsЊ *
 
я2ь
E__inference_dense_layer_call_and_return_all_conditional_losses_545163Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
а2Э
&__inference_dense_layer_call_fn_545152Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
ё2ю
G__inference_dense_1_layer_call_and_return_all_conditional_losses_545194Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
в2Я
(__inference_dense_1_layer_call_fn_545183Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
3B1
$__inference_signature_wrapper_544982input_1
м2й
-__inference_dense_activity_regularizer_544715Ї
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
ы2ш
A__inference_dense_layer_call_and_return_conditional_losses_545143Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
о2л
/__inference_dense_1_activity_regularizer_544739Ї
В
FullArgSpec
args
jself
jx
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *Ђ
	
э2ъ
C__inference_dense_1_layer_call_and_return_conditional_losses_545174Ђ
В
FullArgSpec
args
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs 
kwonlydefaults
 
annotationsЊ *
 
!__inference__wrapped_model_544691m	
1Ђ.
'Ђ$
"
input_1џџџџџџџџџш
Њ "2Њ/
-
dense_1"
dense_1џџџџџџџџџ\
/__inference_dense_1_activity_regularizer_544739)Ђ
Ђ

self
Њ " З
G__inference_dense_1_layer_call_and_return_all_conditional_losses_545194l0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "4Ђ1

0џџџџџџџџџ

	
1/0 Ѕ
C__inference_dense_1_layer_call_and_return_conditional_losses_545174^0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "&Ђ#

0џџџџџџџџџ
 }
(__inference_dense_1_layer_call_fn_545183Q0Ђ-
&Ђ#
!
inputsџџџџџџџџџ
Њ "џџџџџџџџџZ
-__inference_dense_activity_regularizer_544715)Ђ
Ђ

self
Њ " Е
E__inference_dense_layer_call_and_return_all_conditional_losses_545163l	
0Ђ-
&Ђ#
!
inputsџџџџџџџџџш
Њ "4Ђ1

0џџџџџџџџџ

	
1/0 Ѓ
A__inference_dense_layer_call_and_return_conditional_losses_545143^	
0Ђ-
&Ђ#
!
inputsџџџџџџџџџш
Њ "&Ђ#

0џџџџџџџџџ
 {
&__inference_dense_layer_call_fn_545152Q	
0Ђ-
&Ђ#
!
inputsџџџџџџџџџш
Њ "џџџџџџџџџв
H__inference_functional_3_layer_call_and_return_conditional_losses_544840	
9Ђ6
/Ђ,
"
input_1џџџџџџџџџш
p

 
Њ "BЂ?

0џџџџџџџџџ

	
1/0 
	
1/1 в
H__inference_functional_3_layer_call_and_return_conditional_losses_544872	
9Ђ6
/Ђ,
"
input_1џџџџџџџџџш
p 

 
Њ "BЂ?

0џџџџџџџџџ

	
1/0 
	
1/1 б
H__inference_functional_3_layer_call_and_return_conditional_losses_545042	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџш
p

 
Њ "BЂ?

0џџџџџџџџџ

	
1/0 
	
1/1 б
H__inference_functional_3_layer_call_and_return_conditional_losses_545102	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџш
p 

 
Њ "BЂ?

0џџџџџџџџџ

	
1/0 
	
1/1 
-__inference_functional_3_layer_call_fn_544920\	
9Ђ6
/Ђ,
"
input_1џџџџџџџџџш
p

 
Њ "џџџџџџџџџ
-__inference_functional_3_layer_call_fn_544967\	
9Ђ6
/Ђ,
"
input_1џџџџџџџџџш
p 

 
Њ "џџџџџџџџџ
-__inference_functional_3_layer_call_fn_545117[	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџш
p

 
Њ "џџџџџџџџџ
-__inference_functional_3_layer_call_fn_545132[	
8Ђ5
.Ђ+
!
inputsџџџџџџџџџш
p 

 
Њ "џџџџџџџџџ 
$__inference_signature_wrapper_544982x	
<Ђ9
Ђ 
2Њ/
-
input_1"
input_1џџџџџџџџџш"2Њ/
-
dense_1"
dense_1џџџџџџџџџ