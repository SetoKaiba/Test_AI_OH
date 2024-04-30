using System;
using System.Runtime.InteropServices;

public class OHAI
{
    private const int OH_AI_MAX_SHAPE_NUM = 32;

    public struct OH_AI_TensorHandleArray
    {
        public ulong handle_num;
        public IntPtr handle_list;
    }

    public unsafe struct OH_AI_ShapeInfo
    {
        public ulong shape_num;
        public fixed long shape[OH_AI_MAX_SHAPE_NUM];
    }

    public static uint OH_AI_DEVICETYPE_CPU = 0;

    public static uint OH_AI_DEVICETYPE_GPU = 1;

    public static uint OH_AI_DEVICETYPE_KIRIN_NPU = 2;

    // add new type here
    // ohos-only device range: [60, 80)
    public static uint OH_AI_DEVICETYPE_NNRT = 60;
    public static uint OH_AI_DEVICETYPE_INVALID = 100;

    public static uint OH_AI_MODELTYPE_MINDIR = 0;

    // insert new data type here
    public static uint OH_AI_MODELTYPE_INVALID = 0xFFFFFFFF;

    // Helper
    [DllImport("AI")]
    public static extern void Tuanjie_OH_AI_PrintArray(OH_AI_TensorHandleArray array,
        [MarshalAs(UnmanagedType.LPStr)] string prefix);

    // Context
    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_ContextCreate();

    [DllImport("AI")]
    public static extern void Tuanjie_OH_AI_ContextDestroy(IntPtr context);

    [DllImport("AI")]
    public static extern void Tuanjie_OH_AI_ContextAddDeviceInfo(IntPtr context, IntPtr deviceInfo);


    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_DeviceInfoCreate(uint deviceType);

    // Model
    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_ModelCreate();

    [DllImport("AI")]
    public static extern void Tuanjie_OH_AI_ModelDestroy(IntPtr model);

    [DllImport("AI", CallingConvention = CallingConvention.Cdecl)]
    public static extern int Tuanjie_OH_AI_ModelBuild(IntPtr model, byte[] modelData, ulong dataSize, uint modelType,
        IntPtr modelContext);

    [DllImport("AI")]
    public static extern int Tuanjie_OH_AI_ModelResize(IntPtr model, OH_AI_TensorHandleArray inputs, IntPtr shapeInfos,
        ulong shapeInfoNum);

    [DllImport("AI")]
    public static extern int Tuanjie_OH_AI_ModelPredict(IntPtr model);

    [DllImport("AI")]
    public static extern OH_AI_TensorHandleArray Tuanjie_OH_AI_ModelGetInputs(IntPtr model);

    [DllImport("AI")]
    public static extern OH_AI_TensorHandleArray Tuanjie_OH_AI_ModelGetOutputs(IntPtr model);

    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_ModelGetInputByTensorName(IntPtr model,
        [MarshalAs(UnmanagedType.LPStr)] string tensorName);

    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_ModelGetOutputByTensorName(IntPtr model,
        [MarshalAs(UnmanagedType.LPStr)] string tensorName);

    // Tensor
    [DllImport("AI")]
    public static extern void Tuanjie_OH_AI_TensorSetShape(IntPtr tensor, IntPtr shape, ulong shapeNum);

    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_TensorGetShape(IntPtr tensor, IntPtr shapeNum);

    [DllImport("AI")]
    public static extern void Tuanjie_OH_AI_TensorSetData(IntPtr tensor, IntPtr data);

    [DllImport("AI")]
    public static extern IntPtr Tuanjie_OH_AI_TensorGetData(IntPtr tensor);

    [DllImport("AI")]
    public static extern ulong Tuanjie_OH_AI_TensorGetElementNum(IntPtr tensor);

    [DllImport("AI")]
    public static extern ulong Tuanjie_OH_AI_TensorGetDataSize(IntPtr tensor);
}