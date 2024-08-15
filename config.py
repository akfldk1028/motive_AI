# config.py

AI_MODELS = {
    'RealisticVision': {
        'type': 'checkpoint',
        'path': './models/checkpoints/realisticVisionV60B1_v51HyperVAE.safetensors',
        'app': 'RealisticVision',
        'test': True,
        'sampler': ['ddpm_sde', 'euler_ancestral']
    },
    'Architecture RealMix': {
        'type': 'checkpoint',
        'path': './models/checkpoints/architecturerealmix_v11.safetensors',
        'app': 'ArchitectureRealMix',
        'test': True,

    },
    'ReRender': {
        'type': 'checkpoint',
        'path': './models/checkpoints/raRenderArchitectureRender_v33.safetensors',
        'app': 'ReRender',
        'test': True,
        'sampler': ['ddpm_sde', 'euler_ancestral']
    },
    'Zach illustration': {
        'type': 'lora',
        'path': './models/lora/zarch_illustration.safetensors',
        'app': 'ZachIllustration',
        'baseCheckpoint': 'Architecture RealMix',
        'secondCheckpoint': 'RealisticVision'
    },
    'Modern Interior': {
        'type': 'lora',
        'path': './models/lora/mordernInterior.safetensors',
        'app': 'Lora',
        'url': '',
        'defaultPrompt': '',
    },
    'Building Model': {
        'type': 'lora',
        'path': './models/lora/Manualbuildingmodel.safetensors',
        'app': 'BuildingModel',
        'test': True,
        'baseCheckpoint': 'ReRender',
    },
    'SDXL': {
        'type': 'checkpoint',
        'path': './models/SDXL_base_model',
        'app': 'SDXL'
    }
}

CONTROLNET_MODELS = {
    'SD': {
        'CANNY': {
            'path': './models/controlnet/control_v11p_sd15_canny',
            'preprocessor': 'canny',
            'type': "canny",

        },
        'DEPTH': {
            'path': "./models/controlnet/control_v11f1p_sd15_depth",
            'preprocessor': 'depth',
            'type': "depth",

        },
        'SEGMENTATION': {
            'path': "./models/controlnet/control_v11p_sd15_seg",
            'preprocessor': 'segmentation',
            'type': "segmentation",

        },

        #mlsd normal lineart softedge
        # 다른 SD ControlNet 모델들...
    },
    'SDXL': {
        'CANNY': {
            'path': './models/controlnet/controlnet-depth-sdxl-1.0',
            'preprocessor': 'canny',
            'type': "canny",

        },
        'DEPTH': {
            'path': './models/controlnet/controlnet-depth-sdxl-1.0',
            'preprocessor': 'SDXLdepth',
            'type': "depth",

        },
        # 다른 SDXL ControlNet 모델들...
    }
}


def get_model_info(model_name):
    return AI_MODELS.get(model_name, None)


def get_controlnet_info(controlnet_type):
    return CONTROLNET_MODELS.get(controlnet_type, None)

# architecture Lora : zach  //ddmpp_sde +++
# realistic Lora : zach  //ddmpp_sde    ++


# architecture Lora : xl_film //ddmpp_sde
# realistic Lora : xl_film //ddmpp_sde xxxx Lora 안먹힘


# https://civitai.com/models/574758/fine-line-style-sdxl
# https://civitai.com/models/28112/xsarchitectural-interiordesign-forxslora
# https://civitai.com/models/152745/xsarchi145illustration-style-architecture-interiorv2v2


# 로라(LoRA) 또는 드림부스(Dreambooth) 파인튜닝: AI 모델을 특정 캐릭터에 맞게 미세 조정하여 일관된 이미지를 생성합니다.
# 인페인팅(Inpainting): 원본 이미지의 얼굴 부분을 유지하고 나머지 부분만 새로 생성합니다.
# 포즈 전이: 3D 모델링 기술을 사용해 캐릭터의 포즈만 변경합니다.

# OpenPose:
#
# 카네기 멜론 대학에서 개발한 오픈 소스 실시간 다중 인물 검출 및 포즈 추정 시스템입니다.
# 2D 이미지나 비디오에서 사람의 신체 관절 위치를 감지합니다.
#
#
# DeepPose (Google):
#
# 구글에서 개발한 딥러닝 기반 인체 포즈 추정 모델입니다.
# CNN(합성곱 신경망)을 사용하여 이미지에서 관절 위치를 예측합니다.
#
#
# DensePose (Facebook AI Research):
#
# 2D 이미지에서 3D 표면 모델로의 매핑을 제공합니다.
# 인체의 밀집된 포즈 및 형태 정보를 추정합니다.
#
#
# MediaPipe Pose:
#
# 구글에서 개발한 실시간 포즈 추정 솔루션입니다.
# 모바일 기기에서도 효율적으로 작동하도록 설계되었습니다.
#
#
# AlphaPose:
#
# 다중 인물 포즈 추정을 위한 정확하고 빠른 시스템입니다.
# 복잡한 장면에서도 효과적으로 작동합니다.
#
#
# SMPL (Skinned Multi-Person Linear Model):
#
# 3D 인체 모델을 생성하는 파라메트릭 모델입니다.
# 다양한 체형과 포즈를 표현할 수 있습니다.

# --------------------------------
# Midjourney:
#
# 텍스트 프롬프트를 통해 건축 디자인 컨셉과 이미지를 생성할 수 있습니다.
# 다양한 건축 스타일과 인테리어 디자인 이미지를 만들 수 있습니다.
#
#
# ArchiGAN:
#
# 건축 평면도를 생성하는 데 특화된 GAN(Generative Adversarial Network) 모델입니다.
# 기존 건축 데이터를 학습하여 새로운 평면도 디자인을 제안합니다.
#
#
# CityEngine:
#
# 도시 규모의 3D 모델링을 자동화하는 프로시저럴 모델링 소프트웨어입니다.
# AI 요소를 통합하여 도시 계획 및 건축 디자인을 지원합니다.
#
#
# Spacemaker:
#
# AI를 활용하여 부지 분석, 건물 배치 최적화, 일조권 분석 등을 수행합니다.
# 도시 계획 및 건축 설계 초기 단계에서 의사결정을 지원합니다.
#
#
# TestFit:
#
# 부동산 개발 및 건축 설계를 위한 AI 기반 소프트웨어입니다.
# 건축 규정, 재정적 제약 등을 고려하여 최적의 건물 디자인을 제안합니다.
#
#
# Build.ai:
#
# 건축 설계 프로세스를 자동화하는 AI 플랫폼입니다.
# 건축가의 스케치를 3D 모델로 변환하고, 건축 규정 준수 여부를 확인합니다.


# https://towardsdatascience.com/ai-architecture-f9d78c6958e0
