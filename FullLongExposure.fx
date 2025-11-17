////////////////////////////////////////////////////////////////////////////////////////////////////////
// Motion-Aware Long Exposure.fx
// - Combines RealLongExposure.fx (SirCobra / CobraFX)
// - ReshadeMotionEstimation (Jakob Wapenhensch)
// - LinearMotionBlur (Jakob Wapenhensch)
//
// All source components are licensed under CC BY-NC 4.0.
// Choppiness removed via motion-vector temporal interpolation.
// Crafted by coalaura
////////////////////////////////////////////////////////////////////////////////////////////////////////

#include "ReShade.fxh"
#include "ReShadeUI.fxh"

uniform float timer      < source = "timer";      >;
uniform float frametime  < source = "frametime";  >;
uniform int   framecount < source = "framecount"; >;

texture2D texColor : COLOR;
sampler samplerColor
{
    Texture   = texColor;
    AddressU  = Clamp;
    AddressV  = Clamp;
    MipFilter = Linear;
    MinFilter = Linear;
    MagFilter = Linear;
};

texture texMotionVectors < pooled = false; >
{
    Width  = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RG16F;
};

sampler SamplerMotionVectors
{
    Texture   = texMotionVectors;
    AddressU  = Clamp;
    AddressV  = Clamp;
    MipFilter = Point;
    MinFilter = Point;
    MagFilter = Point;
};

float2 sampleMotion(float2 texcoord)
{
    return tex2D(SamplerMotionVectors, texcoord).rg;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// UI
////////////////////////////////////////////////////////////////////////////////////////////////////////

uniform bool UI_StartExposure <
    ui_label   = "Start Capture";
    ui_category= "Long Exposure";
> = false;

uniform float UI_ExposureDuration <
    ui_label   = "Duration";
    ui_type    = "slider";
    ui_min     = 0.1;
    ui_max     = 120.0;
    ui_step    = 0.1;
    ui_units   = "s";
    ui_category= "Long Exposure";
> = 2.0;

uniform bool UI_ShowProgress <
    ui_label   = "Show Progress";
    ui_category= "Long Exposure";
> = true;

uniform int UI_TemporalSubsamples < __UNIFORM_SLIDER_INT1
    ui_min = 1; ui_max = 16; ui_step = 1;
    ui_label   = "Temporal Subsamples";
    ui_tooltip = "Higher values reduce choppiness on low FPS. Default 4 works for most cases.";
    ui_category= "Long Exposure";
> = 4;

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Motion Estimation
////////////////////////////////////////////////////////////////////////////////////////////////////////

#define BLOCK_SIZE       4
#define BLOCK_SIZE_HALF  2
#define BLOCK_AREA       16

#define ME_PYR_DIVISOR   2
#define ME_PYR_LVL_1_DIV 2
#define ME_PYR_LVL_2_DIV 4
#define ME_PYR_LVL_3_DIV 8
#define ME_PYR_LVL_4_DIV 16
#define ME_PYR_LVL_5_DIV 32
#define ME_PYR_LVL_6_DIV 64
#define ME_PYR_LVL_7_DIV 128

#define M_PI     3.1415926535
#define M_F_R2D (180.f / M_PI)
#define M_F_D2R (1.0 / M_F_R2D)

// Motion estimation defaults
#define ME_LAYER_MAX 0
#define ME_LAYER_MIN 6
#define ME_MAX_ITERATIONS 2
#define ME_SAMPLES_PER_ITERATION 5
#define ME_PYRAMID_UPSCALE_FILTER_RADIUS 4.0
#define ME_PYRAMID_UPSCALE_FILTER_RINGS 3
#define ME_PYRAMID_UPSCALE_FILTER_SAMPLES_PER_RING 5

texture texCur0        < pooled = false; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
texture texLast0       < pooled = false; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA8; };
texture texMotionFilterX< pooled = false; >{ Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };

texture texGCur0       < pooled = false; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
texture texGLast0      < pooled = false; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RG16F; };
texture texMotionCur0  < pooled = false; > { Width = BUFFER_WIDTH; Height = BUFFER_HEIGHT; Format = RGBA16F; };

texture texGCur1       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_1_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_1_DIV; Format = RG16F; };
texture texGLast1      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_1_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_1_DIV; Format = RG16F; };
texture texMotionCur1  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_1_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_1_DIV; Format = RGBA16F; };

texture texGCur2       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_2_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_2_DIV; Format = RG16F; };
texture texGLast2      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_2_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_2_DIV; Format = RG16F; };
texture texMotionCur2  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_2_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_2_DIV; Format = RGBA16F; };

texture texGCur3       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_3_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_3_DIV; Format = RG16F; };
texture texGLast3      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_3_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_3_DIV; Format = RG16F; };
texture texMotionCur3  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_3_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_3_DIV; Format = RGBA16F; };

texture texGCur4       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_4_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_4_DIV; Format = RG16F; };
texture texGLast4      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_4_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_4_DIV; Format = RG16F; };
texture texMotionCur4  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_4_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_4_DIV; Format = RGBA16F; };

texture texGCur5       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_5_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_5_DIV; Format = RG16F; };
texture texGLast5      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_5_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_5_DIV; Format = RG16F; };
texture texMotionCur5  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_5_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_5_DIV; Format = RGBA16F; };

texture texGCur6       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_6_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_6_DIV; Format = RG16F; };
texture texGLast6      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_6_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_6_DIV; Format = RG16F; };
texture texMotionCur6  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_6_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_6_DIV; Format = RGBA16F; };

texture texGCur7       < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_7_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_7_DIV; Format = RG16F; };
texture texGLast7      < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_7_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_7_DIV; Format = RG16F; };
texture texMotionCur7  < pooled = false; > { Width = BUFFER_WIDTH / ME_PYR_LVL_7_DIV; Height = BUFFER_HEIGHT / ME_PYR_LVL_7_DIV; Format = RGBA16F; };

sampler smpCur0          { Texture = texCur0;        AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpLast0         { Texture = texLast0;       AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMotionFilterX { Texture = texMotionFilterX;AddressU = Clamp; AddressV = Clamp; MipFilter = Point;  MinFilter = Point;  MagFilter = Point; };

sampler smpGCur0 { Texture = texGCur0; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast0{ Texture = texGLast0;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur0 { Texture = texMotionCur0; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur1 { Texture = texGCur1; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast1{ Texture = texGLast1;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur1 { Texture = texMotionCur1; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur2 { Texture = texGCur2; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast2{ Texture = texGLast2;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur2 { Texture = texMotionCur2; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur3 { Texture = texGCur3; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast3{ Texture = texGLast3;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur3 { Texture = texMotionCur3; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur4 { Texture = texGCur4; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast4{ Texture = texGLast4;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur4 { Texture = texMotionCur4; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur5 { Texture = texGCur5; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast5{ Texture = texGLast5;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur5 { Texture = texMotionCur5; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur6 { Texture = texGCur6; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast6{ Texture = texGLast6;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur6 { Texture = texMotionCur6; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

sampler smpGCur7 { Texture = texGCur7; AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpGLast7{ Texture = texGLast7;AddressU = Clamp; AddressV = Clamp; MipFilter = Linear; MinFilter = Linear; MagFilter = Point; };
sampler smpMCur7 { Texture = texMotionCur7; AddressU = Clamp; AddressV = Clamp; MipFilter = Point; MinFilter = Point; MagFilter = Point; };

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Motion estimation helper functions
////////////////////////////////////////////////////////////////////////////////////////////////////////

void getBlock(float2 center, out float2 block[BLOCK_AREA], sampler grayIn)
{
    [unroll]
    for (int x = 0; x < BLOCK_SIZE; x++)
    {
        [unroll]
        for (int y = 0; y < BLOCK_SIZE; y++)
        {
            block[(BLOCK_SIZE * y) + x] = tex2Doffset(grayIn, center, int2(x - BLOCK_SIZE_HALF, y - BLOCK_SIZE_HALF)).rg;
        }
    }
}

float2 sampleBlock(int2 coord, float2 block[BLOCK_AREA])
{
    int2 pos = clamp(coord, int2(0, 0), int2(BLOCK_SIZE_HALF - 1, BLOCK_SIZE_HALF - 1));
    return block[(BLOCK_SIZE * coord.y) + coord.x];
}

float getBlockFeatureLevel(float2 block[BLOCK_AREA])
{
    float2 average = 0;
    for (int i = 0; i < BLOCK_AREA; i++)
        average += block[i];
    average /= float(BLOCK_AREA);

    float2 diff = 0;
    for (int i = 0; i < BLOCK_AREA; i++)
        diff += abs(block[i] - average);
    diff /= float(BLOCK_AREA);

    float noise = saturate(diff.x * 2);
    return noise;
}

float perPixelLoss(float2 grayDepthA, float2 grayDepthB)
{
    float2 loss = (grayDepthA - grayDepthB);
    float2 finalLoss = abs(loss);
    return lerp(finalLoss.g, finalLoss.r, 0.75);
}

float blockLoss(float2 blockA[BLOCK_AREA], float2 blockB[BLOCK_AREA])
{
    float summedLosses = 0;
    for (int i = 0; i < BLOCK_AREA; i++)
        summedLosses += perPixelLoss(blockA[i], blockB[i]);
    return (summedLosses / float(BLOCK_AREA));
}

float3 GetNormalVector(float2 texcoord)
{
    float3 offset = float3(ReShade::PixelSize.xy, 0.0);
    float2 posCenter = texcoord.xy;
    float2 posNorth  = posCenter - offset.zy;
    float2 posEast   = posCenter + offset.xz;

    float3 vertCenter = float3(posCenter - 0.5, 1) * ReShade::GetLinearizedDepth(posCenter);
    float3 vertNorth  = float3(posNorth - 0.5,  1) * ReShade::GetLinearizedDepth(posNorth);
    float3 vertEast   = float3(posEast - 0.5,   1) * ReShade::GetLinearizedDepth(posEast);

    return normalize(cross(vertCenter - vertNorth, vertCenter - vertEast)) * 0.5 + 0.5;
}

float4 packGbuffer(float2 unpackedMotion, float featureLevel, float loss)
{
    return float4(unpackedMotion.x, unpackedMotion.y, featureLevel, loss);
}

float2 motionFromGBuffer(float4 gbuffer)
{
    return float2(gbuffer.r, gbuffer.g);
}

float randFloatSeed2(float2 seed)
{
    return frac(sin(dot(seed, float2(12.9898, 78.233))) * 43758.5453) * M_PI;
}

float2 getCircleSampleOffset(const int samplesOnCircle, const float radiusInPixels, const int sampleId, const float angleOffset)
{
    float angleDelta = 360.f / samplesOnCircle;
    float sampleAngle = angleOffset + ((angleDelta * sampleId) * M_F_D2R);
    float2 delta = float2((cos(sampleAngle) * radiusInPixels), (sin(sampleAngle) * radiusInPixels));
    return delta;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Motion Estimation layers
////////////////////////////////////////////////////////////////////////////////////////////////////////

float4 CalcMotionLayer(float2 coord, float2 searchStart, sampler curBuffer, sampler lastBuffer, const int iterations)
{
    float2 localBlock[BLOCK_AREA];
    getBlock(coord, localBlock, curBuffer);

    float2 searchBlock[BLOCK_AREA];
    getBlock(coord + searchStart, searchBlock, lastBuffer);

    float localLoss = blockLoss(localBlock, searchBlock);
    float lowestLoss = localLoss;
    float featuresAtLowestLoss = getBlockFeatureLevel(searchBlock);
    float2 bestMotion = float2(0, 0);
    float2 searchCenter = searchStart;

    float randomValue = randFloatSeed2(coord) * 100;
    randomValue += randFloatSeed2(float2(randomValue, float(framecount % uint(16)))) * 100;

    [loop]
    for (int i = 0; i < iterations; i++)
    {
        randomValue = randFloatSeed2(float2(randomValue, i * 16)) * 100;
        [loop]
        for (int s = 0; s < ME_SAMPLES_PER_ITERATION; s++)
        {
            float2 pixelOffset = (getCircleSampleOffset(ME_SAMPLES_PER_ITERATION, 1, s, randomValue) / tex2Dsize(lastBuffer)) / pow(2, i);
            float2 samplePos = coord + searchCenter + pixelOffset;
            float2 searchBlockB[BLOCK_AREA];
            getBlock(samplePos, searchBlockB, lastBuffer);
            float loss = blockLoss(localBlock, searchBlockB);

            [flatten]
            if (loss < lowestLoss)
            {
                lowestLoss = loss;
                bestMotion = pixelOffset;
                featuresAtLowestLoss = getBlockFeatureLevel(searchBlockB);
            }
        }
        searchCenter += bestMotion;
        bestMotion = float2(0, 0);
    }
    return packGbuffer(searchCenter, featuresAtLowestLoss, lowestLoss);
}

float4 UpscaleMotion(float2 texcoord, sampler curLevelGray, sampler lowLevelGray, sampler lowLevelMotion)
{
    float localDepth = tex2D(curLevelGray, texcoord).g;
    float summedWeights = 0.0;
    float2 summedMotion = float2(0, 0);
    float summedFeatures = 0.0;
    float summedLoss = 0.0;

    float randomValue = randFloatSeed2(texcoord) * 100;
    randomValue += randFloatSeed2(float2(randomValue, float(framecount % uint(16)))) * 100;
    const float distPerCircle = ME_PYRAMID_UPSCALE_FILTER_RADIUS / ME_PYRAMID_UPSCALE_FILTER_RINGS;

    [loop]
    for (int r = 0; r < ME_PYRAMID_UPSCALE_FILTER_RINGS; r++)
    {
        int sampleCount = clamp(ME_PYRAMID_UPSCALE_FILTER_SAMPLES_PER_RING / ((r * 0.5) + 1), 1, ME_PYRAMID_UPSCALE_FILTER_SAMPLES_PER_RING);
        float radius = distPerCircle * (r + 1);
        float circleWeight = 1.0 / (r + 1);
        randomValue += randFloatSeed2(float2(randomValue, r * 10)) * 100;
        [loop]
        for (int i = 0; i < sampleCount; i++)
        {
            float2 samplePos = texcoord + (getCircleSampleOffset(sampleCount, radius, i, randomValue) / tex2Dsize(lowLevelGray));
            float nDepth = tex2D(lowLevelGray, samplePos).r;
            float4 llGBuffer = tex2D(lowLevelMotion, samplePos);
            float loss = llGBuffer.a;
            float features = llGBuffer.b;

            float weightDepth = saturate(1.0 - (abs(nDepth - localDepth) * 1));
            float weightLoss = saturate(1.0 - (loss * 1));
            float weightFeatures = saturate((features * 100));
            float weightLength = saturate(1.0 - (length(motionFromGBuffer(llGBuffer) * 1)));
            float weight = saturate(0.000001 + (weightFeatures * weightLoss * weightDepth * weightLength * circleWeight));

            summedWeights += weight;
            summedMotion += motionFromGBuffer(llGBuffer) * weight;
            summedFeatures += features * weight;
            summedLoss += loss * weight;
        }
    }
    return packGbuffer(summedMotion / summedWeights, summedFeatures / summedWeights, summedLoss / summedWeights);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Motion estimation pixel shaders
////////////////////////////////////////////////////////////////////////////////////////////////////////

void CopyPreviousFramePS(float4 position : SV_Position, float2 texcoord : TEXCOORD,
                         out float4 prevColor : SV_Target0,
                         out float4 prevGray  : SV_Target1)
{
    prevColor = tex2D(smpCur0, texcoord);
    prevGray  = tex2D(smpGCur0, texcoord);
}

float4 StoreCurrentFramePS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2D(ReShade::BackBuffer, texcoord);
}

float2 CurToGrayPS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return float2(dot(tex2D(smpCur0, texcoord).rgb, float3(0.3, 0.5, 0.2)), ReShade::GetLinearizedDepth(texcoord));
}

float4 SaveGray1PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur1, texcoord); }
float4 SaveGray2PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur2, texcoord); }
float4 SaveGray3PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur3, texcoord); }
float4 SaveGray4PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur4, texcoord); }
float4 SaveGray5PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur5, texcoord); }
float4 SaveGray6PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur6, texcoord); }

float4 DownscaleGray1PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur0, texcoord); }
float4 DownscaleGray2PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur1, texcoord); }
float4 DownscaleGray3PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur2, texcoord); }
float4 DownscaleGray4PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur3, texcoord); }
float4 DownscaleGray5PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur4, texcoord); }
float4 DownscaleGray6PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target { return tex2D(smpGCur5, texcoord); }

float4 MotionEstimation6PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 searchStart = float2(0, 0);
    if (ME_LAYER_MIN > 6)
    {
        float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur6, smpGCur7, smpMCur7);
        searchStart = motionFromGBuffer(upscaledLowerLayer);
    }
    float4 curMotionEstimation = 0;
    if (ME_LAYER_MIN > 5)
        curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur6, smpGLast6, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 MotionEstimation5PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 searchStart = float2(0, 0);
    if (ME_LAYER_MIN > 5)
    {
        float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur5, smpGCur6, smpMCur6);
        searchStart = motionFromGBuffer(upscaledLowerLayer);
    }
    float4 curMotionEstimation = 0;
    if (ME_LAYER_MIN > 4)
        curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur5, smpGLast5, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 MotionEstimation4PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 searchStart = float2(0, 0);
    if (ME_LAYER_MIN > 4)
    {
        float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur4, smpGCur5, smpMCur5);
        searchStart = motionFromGBuffer(upscaledLowerLayer);
    }
    float4 curMotionEstimation = 0;
    if (ME_LAYER_MIN > 3)
        curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur4, smpGLast4, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 MotionEstimation3PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float2 searchStart = float2(0, 0);
    if (ME_LAYER_MIN > 3)
    {
        float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur3, smpGCur4, smpMCur4);
        searchStart = motionFromGBuffer(upscaledLowerLayer);
    }
    float4 curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur3, smpGLast3, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 MotionEstimation2PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur2, smpGCur3, smpMCur3);
    float2 searchStart = motionFromGBuffer(upscaledLowerLayer);
    float4 curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur2, smpGLast2, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 MotionEstimation1PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur1, smpGCur2, smpMCur2);
    if (ME_LAYER_MAX > 1)
        return upscaledLowerLayer;
    float2 searchStart = motionFromGBuffer(upscaledLowerLayer);
    float4 curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur1, smpGLast1, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 MotionEstimation0PS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 upscaledLowerLayer = UpscaleMotion(texcoord, smpGCur0, smpGCur1, smpMCur1);
    if (ME_LAYER_MAX > 0)
        return upscaledLowerLayer;
    float2 searchStart = motionFromGBuffer(upscaledLowerLayer);
    float4 curMotionEstimation = CalcMotionLayer(texcoord, searchStart, smpGCur0, smpGLast0, ME_MAX_ITERATIONS);
    return curMotionEstimation;
}

float4 FinalFilterXPS(float4 position : SV_Position, float2 texcoord : TEXCOORD ) : SV_Target { return tex2D(smpMCur0, texcoord); }
float4 FinalFilterYPS(float4 position : SV_Position, float2 texcoord : TEXCOORD ) : SV_Target { return tex2D(smpMotionFilterX, texcoord); }
float2 MotionOutputPS(float4 position : SV_Position, float2 texcoord : TEXCOORD ) : SV_Target { return tex2D(smpMCur0, texcoord).rg; }

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Long Exposure textures
////////////////////////////////////////////////////////////////////////////////////////////////////////

texture TEX_MLXExposure
{
    Width  = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA32F;
};

texture TEX_MLXExposureCopy
{
    Width  = BUFFER_WIDTH;
    Height = BUFFER_HEIGHT;
    Format = RGBA32F;
};

texture TEX_MLXTimer
{
    Width  = 1;
    Height = 1;
    Format = RGBA32F;
};

texture TEX_MLXTimerCopy
{
    Width  = 1;
    Height = 1;
    Format = RGBA32F;
};

sampler2D SAM_MLXExposure     { Texture = TEX_MLXExposure;     AddressU = Clamp; AddressV = Clamp; MagFilter = Point; MinFilter = Point; MipFilter = Point; };
sampler2D SAM_MLXExposureCopy { Texture = TEX_MLXExposureCopy; AddressU = Clamp; AddressV = Clamp; MagFilter = Point; MinFilter = Point; MipFilter = Point; };
sampler2D SAM_MLXTimer        { Texture = TEX_MLXTimer;        AddressU = Clamp; AddressV = Clamp; MagFilter = Point; MinFilter = Point; MipFilter = Point; };
sampler2D SAM_MLXTimerCopy    { Texture = TEX_MLXTimerCopy;    AddressU = Clamp; AddressV = Clamp; MagFilter = Point; MinFilter = Point; MipFilter = Point; };

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Long Exposure helpers
////////////////////////////////////////////////////////////////////////////////////////////////////////

static const float MLX_PI        = 3.1415926535;
static const float MLX_TIME_WRAP = 16777216.0;

float MLX_Mod(float x, float y)
{
    return x - y * floor(x / y);
}

float MLX_WrapDiff(float current, float start)
{
    float diff = current - start;
    if (diff < 0.0)
        diff += MLX_TIME_WRAP;
    return diff;
}

float MLX_Ign(float2 pixel)
{
    return frac(52.9829189 * frac(0.06711056 * pixel.x + 0.00583715 * pixel.y));
}

float3 MLX_SRGBToLinear(float3 c)
{
    float3 le = step(float3(0.04045, 0.04045, 0.04045), c);
    float3 low = c / 12.92;
    float3 high = pow((c + 0.055) / 1.055, 2.4);
    return lerp(low, high, le);
}

float3 MLX_LinearToSRGB(float3 c)
{
    float3 le = step(float3(0.0031308, 0.0031308, 0.0031308), c);
    float3 low = c * 12.92;
    float3 high = 1.055 * pow(abs(c), 1.0 / 2.4) - 0.055;
    return lerp(low, high, le);
}

float3 MLX_DitherLinearToSRGB(float3 c, float2 texcoord)
{
#if (BUFFER_COLOR_BIT_DEPTH == 8)
    float2 pixel = texcoord * float2(BUFFER_WIDTH, BUFFER_HEIGHT);
    float noise = MLX_Ign(pixel);
    float3 srgb = saturate(MLX_LinearToSRGB(c));
    return floor(srgb * 255.0 + noise) / 255.0;
#else
    return MLX_LinearToSRGB(c);
#endif
}

// Temporal interpolation: sample current frame + motion-blurred intermediate samples
float3 MLX_TemporalInterpolatedSample(float2 texcoord)
{
    float3 result = 0.0;
    float2 velocity = sampleMotion(texcoord);

    int substeps = UI_TemporalSubsamples;
    float invSteps = 1.0 / float(substeps);

    [loop]
    for (int i = 0; i < 16; i++)
    {
        if (i >= substeps)
            break;

        float t = (float(i) + 0.5) * invSteps - 0.5;
        float2 samplePos = saturate(texcoord - velocity * t);
        result += tex2Dlod(samplerColor, float4(samplePos, 0.0, 0.0)).rgb;
    }

    result *= invSteps;
    return result;
}

float3 MLX_ShowProgress(float2 texcoord, float3 color, float progress, bool finished)
{
    if (!UI_ShowProgress)
        return color;

    float2 center = float2(0.5, 0.08);
    float aspect = BUFFER_WIDTH / BUFFER_HEIGHT;
    float2 diff = float2((texcoord.x - center.x) * aspect, texcoord.y - center.y);
    float dist = length(diff);
    float ringMask = smoothstep(0.05, 0.048, dist) * smoothstep(0.036, 0.038, dist);
    float angle = (atan2(diff.y, diff.x) + MLX_PI) / (2.0 * MLX_PI);
    float3 ringColor = finished ? float3(0.2, 0.9, 0.4) : float3(0.9, 0.4, 0.2);
    float active = finished ? 1.0 : step(angle, progress);
    color = lerp(color, ringColor, ringMask * active * 0.75);

    if (finished)
    {
        float dotMask = smoothstep(0.015, 0.0, dist);
        color = lerp(color, float3(0.2, 0.9, 0.4), dotMask);
    }
    return color;
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Accumulation & Display
////////////////////////////////////////////////////////////////////////////////////////////////////////

float4 MLX_CopyExposurePS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2D(SAM_MLXExposure, texcoord);
}

float4 MLX_CopyTimerPS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    return tex2D(SAM_MLXTimer, texcoord);
}

float4 MLX_UpdateTimerPS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float currentTime = MLX_Mod(timer, MLX_TIME_WRAP);
    float4 prev = tex2D(SAM_MLXTimerCopy, texcoord);
    float durationMS = UI_ExposureDuration * 1000.0;
    float4 result = prev;

    if (!UI_StartExposure)
    {
        result = float4(currentTime, 0.0, 0.0, 0.0);
    }
    else
    {
        float start = prev.x;
        float samples = prev.y;
        float state = prev.z;

        if (state < 0.5)
        {
            start = currentTime;
            samples = 0.0;
        }

        float elapsed = MLX_WrapDiff(currentTime, start);
        float timeAfterDelay = elapsed;
        bool activeWindow = (timeAfterDelay >= 0.0) && (timeAfterDelay <= durationMS);
        bool finishedWindow = (timeAfterDelay > durationMS);

        if (activeWindow)
            samples += 1.0;

        float newState = finishedWindow ? 2.0 : 1.0;
        result = float4(start, samples, newState, elapsed);
    }
    return result;
}

float4 MLX_AccumulateExposurePS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 timerPrev = tex2D(SAM_MLXTimerCopy, float2(0.5, 0.5));
    float4 timerCur  = tex2D(SAM_MLXTimer, float2(0.5, 0.5));
    float3 history   = tex2D(SAM_MLXExposureCopy, texcoord).rgb;

    bool justStarted = (timerPrev.z < 0.5) && (timerCur.z > 0.5);
    if (!UI_StartExposure || justStarted)
        history = 0.0;

    bool newSample = (timerCur.y > timerPrev.y);
    if (newSample)
    {
        // Use temporal interpolation to smooth motion
        float3 motionSample = MLX_TemporalInterpolatedSample(texcoord);
        float3 linColor = MLX_SRGBToLinear(motionSample);

        // Full exposure accumulation
        history += linColor;
    }

    return float4(history, timerCur.y);
}

float4 MLX_DisplayPS(float4 position : SV_Position, float2 texcoord : TEXCOORD) : SV_Target
{
    float4 timerState = tex2D(SAM_MLXTimer, float2(0.5, 0.5));
    float samples = max(timerState.y, 1.0);
    float3 accum = tex2D(SAM_MLXExposure, texcoord).rgb;
    float3 averaged = accum / samples;

    float3 encoded = MLX_DitherLinearToSRGB(averaged, texcoord * float2(BUFFER_WIDTH, BUFFER_HEIGHT));

    float showExposure = (UI_StartExposure && timerState.y > 0.5) ? 1.0 : 0.0;
    float3 base = tex2D(samplerColor, texcoord).rgb;
    float3 color = lerp(base, encoded, showExposure);

    float durationMS = UI_ExposureDuration * 1000.0;
    float elapsed = timerState.w;
    float progress = durationMS > 0.0 ? saturate(elapsed / durationMS) : 1.0;
    bool finished = (timerState.z >= 1.5);

    color = MLX_ShowProgress(texcoord, color, progress, finished);

    return float4(color, 1.0);
}

////////////////////////////////////////////////////////////////////////////////////////////////////////
// Technique
////////////////////////////////////////////////////////////////////////////////////////////////////////

technique MotionAwareLongExposure
{
    pass CopyPreviousFramePass
    {
        VertexShader = PostProcessVS;
        PixelShader  = CopyPreviousFramePS;
        RenderTarget0 = texLast0;
        RenderTarget1 = texGLast0;
    }

    pass StoreCurrentFramePass
    {
        VertexShader = PostProcessVS;
        PixelShader  = StoreCurrentFramePS;
        RenderTarget = texCur0;
    }

    pass SaveGray1Pass { VertexShader = PostProcessVS; PixelShader = SaveGray1PS; RenderTarget0 = texGLast1; }
    pass SaveGray2Pass { VertexShader = PostProcessVS; PixelShader = SaveGray2PS; RenderTarget0 = texGLast2; }
    pass SaveGray3Pass { VertexShader = PostProcessVS; PixelShader = SaveGray3PS; RenderTarget0 = texGLast3; }
    pass SaveGray4Pass { VertexShader = PostProcessVS; PixelShader = SaveGray4PS; RenderTarget0 = texGLast4; }
    pass SaveGray5Pass { VertexShader = PostProcessVS; PixelShader = SaveGray5PS; RenderTarget0 = texGLast5; }
    pass SaveGray6Pass { VertexShader = PostProcessVS; PixelShader = SaveGray6PS; RenderTarget0 = texGLast6; }

    pass MakeGrayPass       { VertexShader = PostProcessVS; PixelShader = CurToGrayPS;      RenderTarget = texGCur0; }
    pass DownscaleGray1Pass { VertexShader = PostProcessVS; PixelShader = DownscaleGray1PS; RenderTarget = texGCur1; }
    pass DownscaleGray2Pass { VertexShader = PostProcessVS; PixelShader = DownscaleGray2PS; RenderTarget = texGCur2; }
    pass DownscaleGray3Pass { VertexShader = PostProcessVS; PixelShader = DownscaleGray3PS; RenderTarget = texGCur3; }
    pass DownscaleGray4Pass { VertexShader = PostProcessVS; PixelShader = DownscaleGray4PS; RenderTarget = texGCur4; }
    pass DownscaleGray5Pass { VertexShader = PostProcessVS; PixelShader = DownscaleGray5PS; RenderTarget = texGCur5; }
    pass DownscaleGray6Pass { VertexShader = PostProcessVS; PixelShader = DownscaleGray6PS; RenderTarget = texGCur6; }

    pass MotionEstimation6Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation6PS; RenderTarget = texMotionCur6; }
    pass MotionEstimation5Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation5PS; RenderTarget = texMotionCur5; }
    pass MotionEstimation4Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation4PS; RenderTarget = texMotionCur4; }
    pass MotionEstimation3Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation3PS; RenderTarget = texMotionCur3; }
    pass MotionEstimation2Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation2PS; RenderTarget = texMotionCur2; }
    pass MotionEstimation1Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation1PS; RenderTarget = texMotionCur1; }
    pass MotionEstimation0Pass { VertexShader = PostProcessVS; PixelShader = MotionEstimation0PS; RenderTarget = texMotionCur0; }

    pass FinalFilterXPass { VertexShader = PostProcessVS; PixelShader = FinalFilterXPS; RenderTarget = texMotionFilterX; }
    pass FinalFilterYPass { VertexShader = PostProcessVS; PixelShader = FinalFilterYPS; RenderTarget = texMotionCur0; }

    pass MotionOutputPass
    {
        VertexShader = PostProcessVS;
        PixelShader  = MotionOutputPS;
        RenderTarget = texMotionVectors;
    }

    pass CopyExposurePass
    {
        VertexShader = PostProcessVS;
        PixelShader  = MLX_CopyExposurePS;
        RenderTarget = TEX_MLXExposureCopy;
    }

    pass CopyTimerPass
    {
        VertexShader = PostProcessVS;
        PixelShader  = MLX_CopyTimerPS;
        RenderTarget = TEX_MLXTimerCopy;
    }

    pass UpdateTimerPass
    {
        VertexShader = PostProcessVS;
        PixelShader  = MLX_UpdateTimerPS;
        RenderTarget = TEX_MLXTimer;
    }

    pass AccumulateExposurePass
    {
        VertexShader = PostProcessVS;
        PixelShader  = MLX_AccumulateExposurePS;
        RenderTarget = TEX_MLXExposure;
    }

    pass DisplayExposurePass
    {
        VertexShader = PostProcessVS;
        PixelShader  = MLX_DisplayPS;
    }
}