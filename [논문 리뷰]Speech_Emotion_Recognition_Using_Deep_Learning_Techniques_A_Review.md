## Speech Emotion Recognition Using Deep Learning Techniques: A Review
---
### ABSTRACT 초록

감정 인식은 Human-Computer Interaction(HCI)에서 중요한 요소이다.  
speech emotion recognition(SER) 분야에서 다양한 기술들이 사용되고 있는데 딥러닝 기술은 전통적인 SER 기술을 대체할 기술로 제안되었다.  
이 논문은 음성 기반의 음성 인식에 대한 딥러닝 기술을 살펴보고 토의하는 내용이다.

---
### 1. INTRODUCTION 소개
음성 인식은 HCI의 작은 분야에서 중요한 요소로 발전했다. 이 기술은 콜센터, 차량운행시스템, 의료 분야 등에서 기계와 음성을 통해 직접 소통할 수 있도록 한다. 그러나 이 기술이 학문연구를 벗어나 실제 생활에 사용되게 하기 위해선 많은 문제들이 있다.  
인간의 감정을 분류하는 모델은 각기다른 감정으로 접근하는 것이 기본 접근법으로 간주된다.
이것은 anger(분노), boredom(지루함), disgust(혐오),
surprise(놀람), fear(두려움), joy(기쁨), happiness(행복), neutral(중립), sadness(슬픔)과 같은 다양한 감정을 사용한다.
arousal, valence, potency를 변수로 사용하는 또 다른 중요한 모델은 3차원 연속공간이다.  
(Arousal (or intensity) is the level of autonomic activation that an event creates, and ranges from calm (or low) to excited (or high). Valence, on the other hand, is the level of pleasantness that an event generates and is defined along a continuum from negative to positive.)  
(Valence, or hedonic tone, is the affective quality referring to the intrinsic attractiveness/"good"-ness (positive valence) or averseness/"bad"-ness (negative valence) of an event, object, or situation.)  
SER에 대한 접근 방식은 주로 특성 추출과 특성 분류로 알려진 두 단계로 구성이 된다.
음성 처리 분야에서 특성 추출 단계는 주로 source-based excitation features, prosodic features, vocal traction factors, and other hybrid features를 사용한다.  
두 번째 단계인 특성 분류에서는 선형 분류와 비선형 분류를 포함한다. 감정 인식을 위하 가장 일반적으로 사용하는 선형 분류는 Bayesian Networks(BN), Maximum Likelihood Principle(MLP), Support Vector Machine(SVM)이 있다.  
speech signal은 non-stationary로 간주되기 때문에 비선형 분류가 SER에서 효과적으로 작동된다고 간주한다.  
SER에 사용 될 수 있는 비선형 분류 모델은 다양한데, 
 Gaussian Mixture Model(GMM), Hidden Markov Model(HMM)이 이에 속한다. 이 모델들은 기본 수준의 특성으로 파생된 정보를 분류하는데 널리 쓰이는 것들이다.  

Linear Predictor Coefficients([LPC](https://sanghyu.tistory.com/41)),   
Mel Energy-spectrum Dynamic Coefficients(MEDC)( MFCC의
DELTA 특징이 MEDC이다.),   
Mel-Frequency Cepstrum Coefficients (MFCC),  
Perceptual Linear Prediction cepstrum coefficients ([PLP](https://www.koreascience.or.kr/article/JAKO201835372350671.pdf)) 
- [다양한 음성 특성 추출 기법 정리 논문](https://www.koreascience.or.kr/article/JAKO201835372350671.pdf)  

와 같은 Energy-based feature는 음성에서 효과적인 감정 인식을 위해 종종 사용된다.  
K-Nearest Neighbor (KNN), Principal Component Analysis (PCA) and Decision trees와 같은 다른 분류 모델도 감정 분류를 위해 적용된다.  
    
---
---
여기서부터  
## *압축요약*

### 논문 제목: Speech Emotion Recognition Using Deep Learning Techniques: A Review
저자: RUHUL AMIN KHALIL, EDWARD JONES, MOHAMMAD INAYATULLAH BABAR,TARIQULLAH JAN, MOHAMMAD HASEEB ZAFAR, AND THAMER ALHUSSAIN
주제: Speech Emotion Recognition(SER)
목적: 130개 이상의 논문을 활용하고 리뷰하여 SER분야에서 부상하고 있는 딥러닝기법에 대해 소개하고 딥러닝기법의 한계와 앞으로의 방향에 대해 제시한다.

### ABSTRACT
음성에서 감정을 인식하는 것은 HCI에서 여렵지만 중요한 요소이다.  
speech emotion recognition(SER, 음성 감정 인식) 학문에서 감정을 추출하기 위해 많은 기술이 사용되어 왔는데 최근 딥러닝 기법이 전통적인 기법에 대한 대안으로 제시되고 있다.  
본 논문에서는 딥러닝 기법의 개요를 설명하고 이러한 방법이 SER에 활용되는 일부 최근 논문에 대해 논의한다.  
리뷰하는 내용은 사용된 데이터베이스, 추출된 감정, 딥러닝의 SER에 대한 기여 및 관련된 제한 사항이다.

### I. INTRODUCTION
음성 인식은 HCI의 한 분야에서 중요한 요소로 발전했다.  
이러한 시스템은 언어 콘텐츠를 직접 입력해 사용하는 대신 직접 음성 상호 작용을 통해 기계와 자연스러운 인터랙션을 촉진하는 것을 목표로 한다.
콜 센터의 대화, 차량 주행 시스템, 의료 분야 등의 애플리케이션에서의 음성 속 감정패턴 활용이나 구어를 활용한 대화 시스템이 이에 포함된다.
인간의 감정을 분류하는 모델은 각기다른 감정으로 접근하는 것이 기본 접근법으로 간주되는데,  
anger(분노), boredom(지루함), disgust(혐오), surprise(놀람), fear(두려움), joy(기쁨), happiness(행복), neutral(중립), sadness(슬픔)과 같은 다양한 감정을 사용한다.  
또 다른 중요한 모델은 arousal(활력), valence(발란스), potency(효력)과 같은 매개 변수를 가진 3차원 연속 공간이다.

SER에 대한 접근 방식은 주로 특성 추출과 특성 분류로 알려진 두 단계로 구성이 된다.  
음성 처리 분야에서 특성 추출 단계는 주로 source-based excitation features, prosodic features, vocal traction factors, and other hybrid features(소스 기반의 흥분 특징, 운율 특징, 음성 견인 요인 및 기타 혼합 특징)를 사용한다. 
두 번째 단계인 특성 분류에서는 선형 분류와 비선형 분류를 포함한다.
감정 인식을 위해 가장 일반적으로 사용하는 선형 분류는 Bayesian Networks(BN), Maximum Likelihood Principle(MLP), Support Vector Machine(SVM)이 있다.  
speech signal은 non-stationary로 간주되기 때문에 비선형 분류가 SER에서 효과적으로 작동된다고 간주한다.  
SER에 사용 될 수 있는 비선형 분류 모델은 다양한데,  
Gaussian Mixture Model(GMM), Hidden Markov Model(HMM)이 이에 속한다. 이 모델들은 기본 수준의 특성으로 파생된 정보를 분류하는데 널리 쓰이는 것들이다.  

아래와 같은 에너지 기반의 특성 추출 기법이 음성에서 효과적인 감정 인식을 위해 종종 사용된다.  
Linear Predictor Coefficients([LPC](https://sanghyu.tistory.com/41)),   
Mel Energy-spectrum Dynamic Coefficients(MEDC)( MFCC의
DELTA 특징이 MEDC이다.),   
Mel-Frequency Cepstrum Coefficients (MFCC),  
Perceptual Linear Prediction cepstrum coefficients ([PLP](https://www.koreascience.or.kr/article/JAKO201835372350671.pdf)) 
- [다양한 음성 특성 추출 기법 정리 논문](https://www.koreascience.or.kr/article/JAKO201835372350671.pdf) 

KNN(K-Nearest Neighbor), PCA(Principal Component Analysis) 및 Decision tree를 포함한 다른 분류 모델도 감정 인식을 위해 적용된다.

*딥러닝기법은 새로운 분야로 최근 주목을 받고있다.*  
딥러닝 기법의 이점: 수동적인 특성 추출 및 튜닝 없이 복잡한 구조와 특징을 감지하는 기능, raw data에서 낮은 수준의 기능 추출, label이 없는 데이터를 처리하는 능력 등

- Deep Neural Networks (DNNs):  
    입력층과 출력층 사이에 하나 이상의 히든레이어로 구성된 구조를 기반으로 한다.  
- Convolutional Neural Networks(CNN):  
    이미지 및 비디오 처리에 효율적인 결과를 제공한다.  
    고차원 입력 데이터에서 특징을 추출하는 장점이 있지만 작은 변화와 왜곡에서도 특징을 학습하므로 대규모 데이터베이스가 필요하다.

- Recurrent Neural Networks(RNNs), Long Short-Term Memory (LSTM) :  
    자연어 처리(NLP)나 SER와 같은 음성 기반 분류에 훨씬 효과적이다.  


본 논문의 구성
Section.2: 전통적인 기법을 활용한 SER 에 관련된 배경 소개
Section.3: 딥러닝기법의 필요성 검토
Section.4: 다양한 딥러닝기법에 대한 논의와 향후 방향, , SER에 딥러닝기법을 활용한 논문들 요약
Section.5: 결론

---

### II. TRADITIONAL TECHNIQUES FOR SER : 전통적인 SER 기법  
디지털화된 음성 기반의 감정 인식 시스템은 음성 신호 전처리, 특징 추출 및 분류의 세 가지 기본 요소로 구성된다. #그래프 삽입
- 음성 신호 전처리: 디노이즈와 같은 전처리로 음성에서 의미 있는 단위를 추출한다.
- 특징 추출: 음성에서 사용할 수 있는 특징을 추출하고 식별한다.
- 분류: 추출된 특성으로 감성을 매핑한다. 주로 GMM, HMM 등의 분류모델이 활용된다.

더욱 자세한 설명  
A. ENHANCEMENT OF INPUT SPEECH DATA IN SER : SER에서의 인풋 음성 데이터 향상  
    인풋으로 들어온 데이터는 노이즈로 인해 오염된 경우가 많다. 이럴 경우 특성 추출 및 분류의 정확도가 떨어진다. 하여 음성 데이터 향상은 SER의 중요한 단계이다. 이 단계에서 감정적인 특성만 유지하여 디노이즈를 수행한다.

B. FEATURE EXTRACTION AND SELECTION IN SER : SER에서의 특성 추출과 선택  
    음성 데이터 향상 후 음성 신호는 segments라고 하는 의미있는 단위로 특징지어진다. 다양한 특징을 추출하고 그 정보를 기반으로 추출된 특징을 다양한 분류로 나눈다.
    그 유형은 아래와 같다.
    1) energy, formants, pitch와 같은 단기간 특성에 기초한 단기 분류  
    2) 평균과 표준 편차와 같은 장기 분류로 운율적 특징 중 intensity(강도), pitch(음조), rate of spoken words(구어 속도)를 중요하게 활용한다.  
    이에 기초한 몇 가지 특성을 표 2에 정리하였다.
    #표 삽입

C. MEASURES FOR ACOUSTICS IN SER: SER의 음향 측정  
    음성 매개 변수와 감정 인식과의 관계에 대해 설명한다. intensity(강도), pitch(음조), rate of spoken words(구어 속도) 등은 음성과 관변된 매개변수로 흔히 간주된다.  
    종종, intensity와 pitch는 활성화와 상관관계가 있기 때문에, intensity의 값은 높은 pitch와 함께 증가하고 낮은 pitch와 함께 낮아진다.  
    음향 변수에서 감정으로의 매핑에 영향을 미치는 요소로는 화자가 연기하고 있는지, 스피커 변형이 높은지, 개인의 기분이나 성격 등이 있다.

HCI에서 감정은 약하게 표현되고 혼합되며 서로 구별하기가 어렵다. 학문적으로는 배우들의 다소 과장된 감정 표현이 들어간 음성이 사용된다.  
핵심적인 감정을 arousal축 과 valence축를 이용한 공간 개의 영역별로 설명하자면 그림2와 같다. #그림2 삽입  
- arousal: 차분함이나 흥분의 강도를 나타낸다.  
- valance: 긍정적, 부정적인 감정을 나타낸다.  



