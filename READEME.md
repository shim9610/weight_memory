# Weight Memory: Architecture for Inferring Dataset Characteristics from Learned Weights

## Overview

This repository explores a novel concept in which the learned weights of deep learning models are interpreted as integrative memory structures. Specifically, it investigates whether transformer-based autoencoders' learned parameters can represent implicitly encoded logical and semantic structures.

By training Vision Transformer (ViT) based autoencoders (such as ViT-MAE) on various datasets, this project aims to:

1. **Demonstrate** that learned weights can represent structured logical or semantic concepts.
2. **Develop** methodologies to quantify and extract these implicit representations.
3. **Explore** implications for cognitive science, neuroscience, and artificial intelligence.

## Current Progress

* [x] Implementation and basic evaluation of ViT-MAE model classification on CIFAR-10.
* [ ] Design of quantitative validation experiments for logical structure representation in learned weights.
* [ ] Drafting mathematical framework for transforming learned weights into logical vector spaces.

## Key Objectives

* **Train and fine-tune** transformer-based autoencoder models (initially ViT-MAE based architectures).
* Conduct **systematic experiments** to validate encoding of specific logical structures or procedural reasoning capabilities within learned weights.
* **Propose and validate** mathematical frameworks for converting learned weight spaces into interpretable semantic or logical vector spaces.

## Current Implementation

* Utilizing Hugging Face Transformers, specifically ViT-MAE models fine-tuned for classification tasks on datasets such as CIFAR-10.
* Robust training loops with validation and best model saving strategies included in the training scripts.
* Optimized GPU usage through proper data loading and CUDA configurations.

## Future Research Directions

* Develop comprehensive mathematical models linking deep learning weights to logical structures.
* Collaborate with mathematicians and neuroscientists to refine theoretical frameworks.
* Investigate potential applications in explainable AI, cognitive modeling, and the philosophical foundations of logic.

## Setup and Usage

Clone the repository and set up the environment:

```bash
git clone https://github.com/shim9610/weight_memory.git
cd weight_memory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Run the training script:

```bash
python main.py
```

## License

This repository is licensed under Apache License 2.0, based on Hugging Face Transformers ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers)). All modifications and additions are also provided under the same Apache License 2.0.

---
# Weight Memory: 학습된 가중치 기반 데이터 특성 추론 아키텍쳐 연구

## 개요

이 저장소는 딥러닝 모델의 학습된 가중치가 통합적 메모리 구조로 해석될 수 있다는 새로운 개념을 탐구합니다. 구체적으로 트랜스포머 기반 오토인코더의 학습된 가중치가 학습 중에 내재적으로 암호화된 논리적 및 의미론적 구조를 표현할 수 있는지 조사합니다.

Vision Transformer(ViT) 기반의 오토인코더(ViT-MAE 등)를 다양한 데이터셋에서 학습하여 다음 목표를 달성합니다:

1. 학습된 가중치가 특정 구조화된 논리나 의미적 개념을 표현할 수 있음을 **입증**합니다.
2. 이 암묵적 표현을 정량화하고 추출하는 방법론을 **개발**합니다.
3. 인지과학, 신경과학, 인공지능 분야에 대한 이러한 표현의 시사점을 **탐구**합니다.

## 현재 진행 단계

* [x] CIFAR-10을 활용한 ViT-MAE 모델의 기본적인 분류 학습 및 평가 코드 구현
* [ ] 학습된 가중치 내 논리적 구조 표현 능력의 정량적 검증 실험 설계
* [ ] 학습된 가중치를 논리 벡터 공간으로 변환하는 수학적 프레임워크 초안 작성

## 주요 목표

* 트랜스포머 기반 오토인코더 모델(ViT-MAE 기반 아키텍처 등)의 **학습과 파인튜닝**.
* 학습된 가중치 내에서 특정 논리 구조나 절차적 추론 능력이 인코딩될 수 있는지 검증하는 **체계적 실험 수행**.
* 학습된 가중치 공간을 해석 가능한 의미론적 또는 논리적 벡터 공간으로 변환하는 수학적 프레임워크를 **제안하고 검증**.

## 현재 구현

* Hugging Face Transformers를 활용하여 CIFAR-10과 같은 데이터셋에서 분류 작업을 위해 ViT-MAE 모델을 파인튜닝합니다.
* 검증 및 최적 모델 저장 전략이 포함된 견고한 학습 루프가 모델 학습 스크립트에 포함되어 있습니다.
* 적절한 데이터 로딩과 CUDA 설정을 통해 효율적인 GPU 활용을 최적화했습니다.

## 향후 연구 방향

* 딥러닝 가중치를 논리적 구조와 연결하는 포괄적인 수학 모델 개발.
* 수학자 및 신경과학자와의 협력을 통해 이론적 프레임워크를 개선합니다.
* 설명 가능한 AI, 인지 모델링 및 논리의 철학적 기반에 대한 잠재적 응용 분야를 탐구합니다.

## 설정 및 사용법

저장소를 클론하고 환경을 설정합니다:

```bash
git clone https://github.com/shim9610/weight_memory.git
cd weight_memory
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

학습 스크립트 실행:

```bash
python main.py
```

## 라이센스

본 저장소는 Hugging Face Transformers ([https://github.com/huggingface/transformers](https://github.com/huggingface/transformers))를 기반으로 Apache License 2.0에 따라 제공됩니다. 모든 수정 및 추가된 내용 또한 동일한 Apache License 2.0 라이센스를 따릅니다.

