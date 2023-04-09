//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
// Laura's edge detect shader
// Copyright © 2023 coalaura
//+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

#include "ReShade.fxh"

uniform float EdgeStrength <
	ui_type = "slider";
	ui_label = "Edge Strentgh";
    ui_tooltip = "Adjusts the how high the difference between pixels should be to be considered an edge.";
	ui_min = 0.1;
	ui_max = 15.0;
	ui_step = 0.1;
> = 1.0;

uniform int SampleSize <
	ui_type = "slider";
	ui_label = "Sample Size";
    ui_tooltip = "How much of the area should be sampled.";
	ui_min = 1;
	ui_max = 10;
	ui_step = 1;
> = 1;

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

void EdgeDetectPS(float4 vpos : SV_Position, float2 texcoord : TEXCOORD, out float4 res : SV_Target0)
{
    float depth = tex2Dlod(ReShade::DepthBuffer, float4(texcoord, 0, 0)).x;

	float diff = 0.0;

	for (int x = -SampleSize; x <= SampleSize; x++) {
		for (int y = -SampleSize; y <= SampleSize; y++) {
			if (x == 0 && y == 0) continue;

			float d = tex2Dlod(ReShade::DepthBuffer, float4(texcoord + float2(BUFFER_RCP_WIDTH * x, BUFFER_RCP_HEIGHT * y), 0, 0)).x;

			diff += abs(depth - d);
		}
	}

    diff = diff / (SampleSize * SampleSize);

	float4 color = tex2D(ReShade::BackBuffer, texcoord);

	float edge = EdgeStrength / 1000.0;

	if (diff >= edge) {
		float strength = 1.0 + (diff / edge);

		color.x = clamp(color.x * strength, 0, 1);
		color.y = clamp(color.y * strength, 0, 1);
		color.z = clamp(color.z * strength, 0, 1);
	}

	res.xyz = color.xyz;
	res.w = 1.0;
}

//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
//
//++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

technique EdgeDetect
{
	pass EdgeDetect_Apply
	{
		VertexShader = PostProcessVS;
		PixelShader = EdgeDetectPS;
	}
}
