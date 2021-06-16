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

### ABSTRACT 초록
음성에서 감정을 인식하는 것은 HCI에서 여렵지만 중요한 요소이다.  
speech emotion recognition(SER, 음성 감정 인식) 학문에서 감정을 추출하기 위해 많은 기술이 사용되어 왔는데 최근 딥러닝 기법이 전통적인 기법에 대한 대안으로 제시되고 있다.

