//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Laura's adaptive brightness
// Copyright © 2023 coalaura
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ReShade.fxh"

uniform bool DoBrightness <
	ui_label = "Apply Brightness";
> = false;

uniform float BrightnessContrast <
	ui_type = "slider";
	ui_label = "Contrast Adjustment";
    ui_tooltip = "Adjusts the contrast of the brightened areas.";
	ui_min = 0.0;
	ui_max = 0.15;
	ui_step = 0.002;
> = 0.065;

uniform float BrightnessLevel <
	ui_type = "slider";
	ui_label = "Brightness Level";
	ui_min = 0.0;
	ui_max = 15.0;
	ui_step = 0.01;
> = 8.0;

uniform float BrightnessCurve <
	ui_type = "slider";
	ui_label = "Curve Steepness";
    ui_tooltip = "Specifies how dark an area needs to be before it is brightened.";
	ui_min = 2.0;
	ui_max = 30.0;
	ui_step = 0.1;
> = 14.0;

uniform int BrightnessIterations <
	ui_type = "slider";
	ui_label = "Iterations";
    ui_tooltip = "Specifies how often the brightness level should be applied.";
	ui_min = 1;
	ui_max = 10;
	ui_step = 1;
> = 1;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void _brighten(inout float4 color, float amount)
{
	float gray = 1.0 - (0.2989 * color.x + 0.5870 * color.y + 0.1140 * color.z);

	amount *= 1.0 - sqrt(1.0 - pow(abs(gray), BrightnessCurve));

	amount += 1.0;

	float contrast = 1.0 + (amount * BrightnessContrast);
	float intercept = 0.5 * (1.0 - contrast);

	color.x = clamp((color.x * amount) * contrast + intercept, 0.0, 1.0);
	color.y = clamp((color.y * amount) * contrast + intercept, 0.0, 1.0);
	color.z = clamp((color.z * amount) * contrast + intercept, 0.0, 1.0);
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void AdaptiveBrightnessPS(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 res : SV_Target0)
{
    float4 color = tex2D(ReShade::BackBuffer, texcoord);

    if (DoBrightness)
    {
		float level = BrightnessLevel / BrightnessIterations;

        for (int i = 0; i < BrightnessIterations; i++)
        {
            _brighten(color, level);
        }
    }

    res.xyz = color.xyz;
	res.w = 1.0;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

technique AdaptiveBrightness
{
	pass AdaptiveBrightness_Apply
	{
		VertexShader = PostProcessVS;
		PixelShader = AdaptiveBrightnessPS;
	}
}
