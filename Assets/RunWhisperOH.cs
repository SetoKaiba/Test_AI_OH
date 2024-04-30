using System;
using System.Collections.Generic;
using UnityEngine;
using Unity.Sentis;
using Newtonsoft.Json;
using System.Text;
using Unity.Collections.LowLevel.Unsafe;

/*
 *              Whisper Inference Code
 *              ======================
 *
 *  Put this script on the Main Camera
 *
 *  In Assets/StreamingAssets put:
 *
 *  AudioDecoder_Tiny.sentis
 *  AudioEncoder_Tiny.sentis
 *  LogMelSepctro.sentis
 *  vocab.json
 *
 *  Drag a 30s 16khz mono uncompressed audioclip into the audioClip field.
 *
 *  Install package com.unity.nuget.newtonsoft-json from packagemanger
 *  Install package com.unity.sentis
 *
 */


public class RunWhisperOH : MonoBehaviour
{
    // Link your audioclip here. Format must be 16Hz mono non-compressed.
    public AudioClip audioClip;
    public TextAsset argmaxAsset;
    public TextAsset decoderAsset;
    public TextAsset encoderAsset;
    public TextAsset spectroAsset;
    public TextAsset vocabJson;

    public IntPtr context;
    public IntPtr argmax;
    public IntPtr decoder;
    public IntPtr encoder;
    public IntPtr spectro;

    // This is how many tokens you want. It can be adjusted.
    const int maxTokens = 100;

    //Special tokens see added tokens file for details
    const int END_OF_TEXT = 50257;
    const int START_OF_TRANSCRIPT = 50258;
    const int ENGLISH = 50259;
    const int CHINESE = 50260;
    const int GERMAN = 50261;
    const int FRENCH = 50265;
    const int TRANSCRIBE = 50359; //for speech-to-text in specified language
    const int TRANSLATE = 50358; //for speech-to-text then translate to English
    const int NO_TIME_STAMPS = 50363;
    const int START_TIME = 50364;

    int numSamples;
    float[] data;
    string[] tokens;

    int currentToken = 0;
    int[] outputTokens = new int[maxTokens];

    // Used for special character decoding
    int[] whiteSpaceCharacters = new int[256];

    private IntPtr encodedAudio;
    private long[] encodedAudioShape;

    bool transcribe = false;
    string outputString = "";

    // Maximum size of audioClip (30s at 16kHz)
    const int maxSamples = 30 * 16000;

    void Start()
    {
        SetupWhiteSpaceShifts();

        GetTokens();

        int result;

        context = OHAI.Tuanjie_OH_AI_ContextCreate();
        var deviceInfo = OHAI.Tuanjie_OH_AI_DeviceInfoCreate(OHAI.OH_AI_DEVICETYPE_CPU);
        OHAI.Tuanjie_OH_AI_ContextAddDeviceInfo(context, deviceInfo);
        argmax = OHAI.Tuanjie_OH_AI_ModelCreate();
        decoder = OHAI.Tuanjie_OH_AI_ModelCreate();
        encoder = OHAI.Tuanjie_OH_AI_ModelCreate();
        spectro = OHAI.Tuanjie_OH_AI_ModelCreate();
        result = OHAI.Tuanjie_OH_AI_ModelBuild(argmax, argmaxAsset.bytes, (ulong)argmaxAsset.bytes.Length,
            OHAI.OH_AI_MODELTYPE_MINDIR, context);
        if (result != 0)
        {
            Debug.Log("argmax model build failed");
        }

        result = OHAI.Tuanjie_OH_AI_ModelBuild(decoder, decoderAsset.bytes, (ulong)decoderAsset.bytes.Length,
            OHAI.OH_AI_MODELTYPE_MINDIR, context);
        if (result != 0)
        {
            Debug.Log("decoder model build failed");
        }

        result = OHAI.Tuanjie_OH_AI_ModelBuild(encoder, encoderAsset.bytes, (ulong)encoderAsset.bytes.Length,
            OHAI.OH_AI_MODELTYPE_MINDIR, context);
        if (result != 0)
        {
            Debug.Log("encoder model build failed");
        }

        result = OHAI.Tuanjie_OH_AI_ModelBuild(spectro, spectroAsset.bytes, (ulong)spectroAsset.bytes.Length,
            OHAI.OH_AI_MODELTYPE_MINDIR, context);
        if (result != 0)
        {
            Debug.Log("logmelspectro model build failed");
        }

        unsafe
        {
            var encoderShapeInfoArray = new OHAI.OH_AI_ShapeInfo[1];
            encoderShapeInfoArray[0].shape_num = 3;
            encoderShapeInfoArray[0].shape[0] = 1;
            encoderShapeInfoArray[0].shape[1] = 80;
            encoderShapeInfoArray[0].shape[2] = 3000;

            var encoderInputs = OHAI.Tuanjie_OH_AI_ModelGetInputs(encoder);
            fixed (OHAI.OH_AI_ShapeInfo* shapeInfos = encoderShapeInfoArray)
            {
                result = OHAI.Tuanjie_OH_AI_ModelResize(encoder, encoderInputs, (IntPtr)shapeInfos, 1);
            }

            if (result != 0)
            {
                Debug.Log("Encoder model resize failed");
                return;
            }

            var decoderShapeInfoArray = new OHAI.OH_AI_ShapeInfo[2];
            decoderShapeInfoArray[0].shape_num = 2;
            decoderShapeInfoArray[0].shape[0] = 1;
            decoderShapeInfoArray[0].shape[1] = outputTokens.Length;
            decoderShapeInfoArray[1].shape_num = 3;
            decoderShapeInfoArray[1].shape[0] = 1;
            decoderShapeInfoArray[1].shape[1] = 1500;
            decoderShapeInfoArray[1].shape[2] = 384;

            var decoderInputs = OHAI.Tuanjie_OH_AI_ModelGetInputs(decoder);
            fixed (OHAI.OH_AI_ShapeInfo* shapeInfos = decoderShapeInfoArray)
            {
                result = OHAI.Tuanjie_OH_AI_ModelResize(decoder, decoderInputs, (IntPtr)shapeInfos, 2);
            }

            if (result != 0)
            {
                Debug.Log("Decoder model resize failed");
            }

            var argmaxShapeInfoArray = new OHAI.OH_AI_ShapeInfo[1];
            argmaxShapeInfoArray[0].shape_num = 3;
            argmaxShapeInfoArray[0].shape[0] = 1;
            argmaxShapeInfoArray[0].shape[1] = outputTokens.Length;
            argmaxShapeInfoArray[0].shape[2] = 51865;

            var argmaxInputs = OHAI.Tuanjie_OH_AI_ModelGetInputs(argmax);
            fixed (OHAI.OH_AI_ShapeInfo* shapeInfos = argmaxShapeInfoArray)
            {
                result = OHAI.Tuanjie_OH_AI_ModelResize(argmax, argmaxInputs, (IntPtr)shapeInfos, 1);
            }

            if (result != 0)
            {
                Debug.Log("Argmax model resize failed");
            }
        }

        outputTokens[0] = START_OF_TRANSCRIPT;
        outputTokens[1] = CHINESE; // GERMAN;//FRENCH;//
        outputTokens[2] = TRANSCRIBE; //TRANSLATE;//
        outputTokens[3] = NO_TIME_STAMPS; // START_TIME;//
        currentToken = 3;

        LoadAudio();
        transcribe = EncodeAudio();
    }

    void LoadAudio()
    {
        if (audioClip.frequency != 16000)
        {
            Debug.Log($"The audio clip should have frequency 16kHz. It has frequency {audioClip.frequency / 1000f}kHz");
            return;
        }

        numSamples = audioClip.samples;

        if (numSamples > maxSamples)
        {
            Debug.Log(
                $"The AudioClip is too long. It must be less than 30 seconds. This clip is {numSamples / audioClip.frequency} seconds.");
            return;
        }

        data = new float[maxSamples];
        numSamples = maxSamples;
        //We will get a warning here if data.length is larger than audio length but that is OK
        audioClip.GetData(data, 0);
    }


    void GetTokens()
    {
        var vocab = JsonConvert.DeserializeObject<Dictionary<string, int>>(vocabJson.text);
        tokens = new string[vocab.Count];
        foreach (var item in vocab)
        {
            tokens[item.Value] = item.Key;
        }
    }

    void DebugTensorFloat(IntPtr tensor, string tensorName)
    {
        unsafe
        {
            var tensorElementNum = OHAI.Tuanjie_OH_AI_TensorGetElementNum(tensor);
            var tensorDataSize = OHAI.Tuanjie_OH_AI_TensorGetDataSize(tensor);
            var tensorData = OHAI.Tuanjie_OH_AI_TensorGetData(tensor);
            var tensorArray = new float[tensorElementNum];
            fixed (float* addr = tensorArray)
            {
                UnsafeUtility.MemCpy(addr, tensorData.ToPointer(), (long)tensorDataSize);
            }

            ulong shapeNum;
            var shape = OHAI.Tuanjie_OH_AI_TensorGetShape(tensor, (IntPtr)(&shapeNum));
            var shapeArray = new long[shapeNum];
            fixed (long* shapeAddr = shapeArray)
            {
                UnsafeUtility.MemCpy(shapeAddr, shape.ToPointer(), (long)shapeNum * sizeof(long));
            }

            Debug.Log($"{tensorName} elementNum:{tensorElementNum}");
            Debug.Log($"{tensorName} dataSize:{tensorDataSize}");
            Debug.Log($"{tensorName} data:{string.Join(",", tensorArray)}");
            Debug.Log($"{tensorName} shapeNum:{shapeNum}");
            Debug.Log($"{tensorName} shape:{string.Join(",", shapeArray)}");
        }
    }

    void DebugTensorInt(IntPtr tensor, string tensorName)
    {
        unsafe
        {
            var tensorElementNum = OHAI.Tuanjie_OH_AI_TensorGetElementNum(tensor);
            var tensorDataSize = OHAI.Tuanjie_OH_AI_TensorGetDataSize(tensor);
            var tensorData = OHAI.Tuanjie_OH_AI_TensorGetData(tensor);
            var tensorArray = new int[tensorElementNum];
            fixed (int* addr = tensorArray)
            {
                UnsafeUtility.MemCpy(addr, tensorData.ToPointer(), (long)tensorDataSize);
            }

            ulong shapeNum;
            var shape = OHAI.Tuanjie_OH_AI_TensorGetShape(tensor, (IntPtr)(&shapeNum));
            var shapeArray = new long[shapeNum];
            fixed (long* shapeAddr = shapeArray)
            {
                UnsafeUtility.MemCpy(shapeAddr, shape.ToPointer(), (long)shapeNum * sizeof(long));
            }

            Debug.Log($"{tensorName} elementNum:{tensorElementNum}");
            Debug.Log($"{tensorName} dataSize:{tensorDataSize}");
            Debug.Log($"{tensorName} data:{string.Join(",", tensorArray)}");
            Debug.Log($"{tensorName} shapeNum:{shapeNum}");
            Debug.Log($"{tensorName} shape:{string.Join(",", shapeArray)}");
        }
    }

    bool EncodeAudio()
    {
        int result;
        unsafe
        {
            var audioTensor = OHAI.Tuanjie_OH_AI_ModelGetInputByTensorName(spectro, "audio");

            fixed (float* ptr = data)
            {
                OHAI.Tuanjie_OH_AI_TensorSetData(audioTensor, (IntPtr)ptr);
            }

            result = OHAI.Tuanjie_OH_AI_ModelPredict(spectro);
            if (result != 0)
            {
                Debug.Log("Spectro predict failed");
                return false;
            }

            var logMelTensor = OHAI.Tuanjie_OH_AI_ModelGetOutputByTensorName(spectro, "log_mel");
            var logMelData = OHAI.Tuanjie_OH_AI_TensorGetData(logMelTensor);

            // DebugTensorFloat(logMelTensor, "logMel");

            var inputFeaturesTensor = OHAI.Tuanjie_OH_AI_ModelGetInputByTensorName(encoder, "input_features");
            OHAI.Tuanjie_OH_AI_TensorSetData(inputFeaturesTensor, logMelData);

            result = OHAI.Tuanjie_OH_AI_ModelPredict(encoder);
            if (result != 0)
            {
                Debug.Log("Encoder predict failed");
                return false;
            }

            var lastHiddenStateTensor = OHAI.Tuanjie_OH_AI_ModelGetOutputByTensorName(encoder, "last_hidden_state");
            encodedAudio = OHAI.Tuanjie_OH_AI_TensorGetData(lastHiddenStateTensor);

            // DebugTensorFloat(lastHiddenStateTensor, "lastHiddenState");

            return true;
        }
    }


    // Update is called once per frame
    void Update()
    {
        if (transcribe && currentToken < outputTokens.Length - 1)
        {
            unsafe
            {
                var encoderHiddenStatesTensor =
                    OHAI.Tuanjie_OH_AI_ModelGetInputByTensorName(decoder, "encoder_hidden_states");
                OHAI.Tuanjie_OH_AI_TensorSetData(encoderHiddenStatesTensor, encodedAudio);
                var inputIdsTensor = OHAI.Tuanjie_OH_AI_ModelGetInputByTensorName(decoder, "input_ids");
                int result;
                fixed (int* tokensSoFar = outputTokens)
                {
                    OHAI.Tuanjie_OH_AI_TensorSetData(inputIdsTensor, (IntPtr)tokensSoFar);
                    result = OHAI.Tuanjie_OH_AI_ModelPredict(decoder);
                    if (result != 0)
                    {
                        Debug.Log("Decoder predict failed");
                        return;
                    }
                }

                var logitsTensor = OHAI.Tuanjie_OH_AI_ModelGetOutputByTensorName(decoder, "logits");
                var logitsData = OHAI.Tuanjie_OH_AI_TensorGetData(logitsTensor);

                // DebugTensorFloat(logitsTensor, "logits");

                var tokensOutTensor = OHAI.Tuanjie_OH_AI_ModelGetInputByTensorName(argmax, "tokens_out");
                OHAI.Tuanjie_OH_AI_TensorSetData(tokensOutTensor, logitsData);
                result = OHAI.Tuanjie_OH_AI_ModelPredict(argmax);
                if (result != 0)
                {
                    Debug.Log("Argmax predict failed");
                }

                var tokensPredictionsTensor =
                    OHAI.Tuanjie_OH_AI_ModelGetOutputByTensorName(argmax, "tokens_predictions");
                var tokensPredictionsData = OHAI.Tuanjie_OH_AI_TensorGetData(tokensPredictionsTensor);

                // DebugTensorInt(tokensPredictionsTensor, "tokensPredictions");

                var tokensPredictionsElementNum = OHAI.Tuanjie_OH_AI_TensorGetElementNum(tokensPredictionsTensor);
                var tokensPredictionsDataSize = OHAI.Tuanjie_OH_AI_TensorGetDataSize(tokensPredictionsTensor);
                var tokensPredictionsArray = new int[tokensPredictionsElementNum];
                fixed (int* tokensPredictionsAddr = tokensPredictionsArray)
                {
                    UnsafeUtility.MemCpy(tokensPredictionsAddr, tokensPredictionsData.ToPointer(),
                        (long)tokensPredictionsDataSize);
                }

                int ID = tokensPredictionsArray[currentToken];
                // Debug.Log($"ID:{ID}");

                outputTokens[++currentToken] = ID;

                if (ID == END_OF_TEXT)
                {
                    transcribe = false;
                }
                else if (ID >= tokens.Length)
                {
                    outputString += $"(time={(ID - START_TIME) * 0.02f})";
                }
                else outputString += GetUnicodeText(tokens[ID]);

                Debug.Log(outputString);
                // transcribe = false;
            }
        }
    }

    // Translates encoded special characters to Unicode
    string GetUnicodeText(string text)
    {
        var bytes = Encoding.GetEncoding("ISO-8859-1").GetBytes(ShiftCharacterDown(text));
        return Encoding.UTF8.GetString(bytes);
    }

    string ShiftCharacterDown(string text)
    {
        string outText = "";
        foreach (char letter in text)
        {
            outText += ((int)letter <= 256) ? letter : (char)whiteSpaceCharacters[(int)(letter - 256)];
        }

        return outText;
    }

    void SetupWhiteSpaceShifts()
    {
        for (int i = 0, n = 0; i < 256; i++)
        {
            if (IsWhiteSpace((char)i)) whiteSpaceCharacters[n++] = i;
        }
    }

    bool IsWhiteSpace(char c)
    {
        return !(('!' <= c && c <= '~') || ('¡' <= c && c <= '¬') || ('®' <= c && c <= 'ÿ'));
    }

    private void OnApplicationQuit()
    {
        if (Input.GetKeyDown(KeyCode.Escape)) Application.Quit();
    }

    private void OnDestroy()
    {
        OHAI.Tuanjie_OH_AI_ModelDestroy(argmax);
        OHAI.Tuanjie_OH_AI_ModelDestroy(decoder);
        OHAI.Tuanjie_OH_AI_ModelDestroy(encoder);
        OHAI.Tuanjie_OH_AI_ModelDestroy(spectro);
        OHAI.Tuanjie_OH_AI_ContextDestroy(context);
    }
}