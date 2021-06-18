## *압축요약*

## 논문 제목: Speech Emotion Recognition Using Deep Learning Techniques: A Review   
<img src = "https://user-images.githubusercontent.com/76423415/122484227-28a63480-d00f-11eb-848a-67066adde24a.png" width="40%">  

[논문링크](https://www.readcube.com/articles/10.1109%2Faccess.2019.2936124)  
저자: RUHUL AMIN KHALIL, EDWARD JONES, MOHAMMAD INAYATULLAH BABAR,TARIQULLAH JAN, MOHAMMAD HASEEB ZAFAR, AND THAMER ALHUSSAIN  
주제: Speech Emotion Recognition(SER)  
목적: 130개 이상의 논문을 활용하고 리뷰하여 SER분야에서 부상하고 있는 딥러닝기법에 대해 소개하고 딥러닝기법의 한계와 앞으로의 방향에 대해 제시한다.  

본 논문의 구성  
Section.2: 전통적인 기법을 활용한 SER 에 관련된 배경 소개  
Section.3: 딥러닝기법의 필요성 검토  
Section.4: 다양한 딥러닝기법에 대한 논의와 향후 방향, , SER에 딥러닝기법을 활용한 논문들 요약  
Section.5: 결론  

---
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
첫 번째인 특성 추출 단계는 주로  
source-based excitation features(소스 기반의 흥분 특징),  
prosodic features( 운율 특징),  
vocal traction factors(음성 견인 요인),  
and other hybrid features(기타 혼합 특징)를 사용한다.  

두 번째 단계인 특성 분류에서는 선형 분류와 비선형 분류를 포함한다.
감정 인식을 위해 가장 일반적으로 사용하는 선형 분류는  
Bayesian Networks(BN), Maximum Likelihood Principle(MLP), Support Vector Machine(SVM)이 있다. 
speech signal은 non-stationary로 간주되기 때문에 비선형 분류가 SER에서 효과적으로 작동된다고 간주한다.  
SER에 사용 될 수 있는 비선형 분류 모델은 다양한데,  
Gaussian Mixture Model(GMM), Hidden Markov Model(HMM)이 이에 속한다.  
이 모델들은 기본 수준의 특성으로 파생된 정보를 분류하는데 널리 쓰이는 것들이다.  

아래와 같은 에너지 기반의 특성 추출 기법이 음성에서 효과적인 감정 인식을 위해 종종 사용된다.  
Linear Predictor Coefficients([LPC](https://sanghyu.tistory.com/41)),   
Mel Energy-spectrum Dynamic Coefficients(MEDC)( MFCC의 DELTA 특징이 MEDC이다.),   
Mel-Frequency Cepstrum Coefficients (MFCC),  
Perceptual Linear Prediction cepstrum coefficients(PLP)  
*참고 [다양한 음성 특성 추출 기법 정리 논문](https://www.koreascience.or.kr/article/JAKO201835372350671.pdf) *

KNN(K-Nearest Neighbor), PCA(Principal Component Analysis) 및 Decision tree를 포함한 다른 분류 모델도 감정 인식을 위해 적용된다.  

딥러닝기법은 새로운 분야로 최근 주목을 받고있다.  
[딥러닝 기법의 이점]  
- 수동적인 특성 추출 및 튜닝 없이 복잡한 구조와 특징을 감지하는 기능
- raw data에서 낮은 수준의 기능 추출
- label이 없는 데이터를 처리하는 능력 등  

본 논문에서는 대표적인 딥러닝 기법을 소개한다.  
- Deep Neural Networks (DNNs):  
    입력층과 출력층 사이에 하나 이상의 히든레이어로 구성된 구조를 기반으로 한다.  
- Convolutional Neural Networks(CNN):  
    이미지 및 비디오 처리에 효율적인 결과를 제공한다.  
    고차원 입력 데이터에서 특징을 추출하는 장점이 있지만 작은 변화와 왜곡에서도 특징을 학습하므로 대규모 데이터베이스가 필요하다.  
- Recurrent Neural Networks(RNNs), Long Short-Term Memory (LSTM) :  
    자연어 처리(NLP)나 SER와 같은 음성 기반 분류에 훨씬 효과적이다.  
    
---

### II. TRADITIONAL TECHNIQUES FOR SER : 전통적인 SER 기법  
디지털화된 음성 기반의 감정 인식 시스템은 음성 신호 전처리, 특징 추출 및 분류의 세 가지 기본 요소로 구성된다.  
![그림3](https://user-images.githubusercontent.com/76423415/122484493-acf8b780-d00f-11eb-913f-2ef5a78d6292.png)

- 음성 신호 전처리: 디노이즈와 같은 전처리로 음성에서 의미 있는 단위를 추출한다.
- 특징 추출: 음성에서 사용할 수 있는 특징을 추출하고 식별한다.
- 분류: 추출된 특성으로 감성을 매핑한다. 주로 GMM, HMM 등의 분류모델이 활용된다.

*더욱 자세한 설명*

A. ENHANCEMENT OF INPUT SPEECH DATA IN SER : SER에서의 인풋 음성 데이터 향상  
인풋으로 들어온 데이터는 노이즈로 인해 오염된 경우가 많다. 이럴 경우 특성 추출 및 분류의 정확도가 떨어진다. 하여 음성 데이터 향상은 SER의 중요한 단계이다. 이 단계에서 감정적인 특성만 유지하여 디노이즈를 수행한다.


B. FEATURE EXTRACTION AND SELECTION IN SER : SER에서의 특성 추출과 선택  
음성 데이터 향상 후 음성 신호는 segments라고 하는 의미있는 단위로 특징지어진다. 다양한 특징을 추출하고 그 정보를 기반으로 추출된 특징을 다양한 분류로 나눈다.  
그 유형은 아래와 같다.  
1) energy, formants, pitch와 같은 단기간 특성에 기초한 단기 분류  
2) 평균과 표준 편차와 같은 장기 분류로 운율적 특징 중 intensity(강도), pitch(음조), rate of spoken words(구어 속도)를 중요하게 활용한다.  
이에 기초한 몇 가지 특성을 표 2에 정리하였다.  
![table2](https://user-images.githubusercontent.com/76423415/122484731-314b3a80-d010-11eb-9dac-5a0a8ef7ff07.PNG)  


C. MEASURES FOR ACOUSTICS IN SER: SER에서의 음향 측정  
음성 매개 변수와 감정 인식과의 관계에 대해 설명한다. intensity(강도), pitch(음조), rate of spoken words(구어 속도) 등은 음성과 관변된 매개변수로 흔히 간주된다.  
종종, intensity와 pitch는 활성화와 상관관계가 있기 때문에, intensity의 값은 높은 pitch와 함께 증가하고 낮은 pitch와 함께 낮아진다.  
음향 변수에서 감정으로의 매핑에 영향을 미치는 요소로는 화자가 연기하고 있는지, 스피커 변형이 높은지, 개인의 기분이나 성격 등이 있다.  

HCI에서 감정은 약하게 표현되고 혼합되며 서로 구별하기가 어렵다. 학문적으로는 배우들의 다소 과장된 감정 표현이 들어간 음성이 사용된다.  
핵심적인 감정을 arousal축 과 valence축를 이용한 공간 개의 영역별로 설명하자면 그림2와 같다.
![그림4](https://user-images.githubusercontent.com/76423415/122484785-52139000-d010-11eb-8c89-6c179c7304e5.png)  
- arousal: 차분함이나 흥분의 강도를 나타낸다.  
- valance: 긍정적, 부정적인 감정을 나타낸다.  

D. CLASSIFICATION OF FEATURES IN SER : SER에서의 특성 분류  
SER에서 사용하는 분류 모델은 크게 2가지로 나눌 수 있다.  
1) 선형 분류: 선형 분포를 가진 다양한 객체의 특징을 기반으로 분류를 수행한다.  
2) 비선형 분류: 객체의 비선형 가중 조합을 개발하는데 사용된다.  
테이블3은 전형적인 SER 방식에서 사용하는 선형, 비선형 모델을 정리한 것이다.  
![table3](https://user-images.githubusercontent.com/76423415/122484819-6a83aa80-d010-11eb-9451-c6fdea5b1a06.PNG)  


E. DATABASES USED FOR SER : SER에서 사용하는 데이터베이스  
데이터베이스는 SER의 성능에 가장 중요한 요소이다.  
SER에서 사용되는 음성 데이터 베이스는 크게 3가지로 나눌 수 있다.  
1) Simulated database: 경험이 풍부한 연기자들에 의해 녹음된 데이터이다. 데이터베이스를 얻는 가장 간단한 방법이다. 음성 데이터베이스의 60% 정도가 이 같은 방법으로 수집된다.  
2) Induced database: 주어진 상황이나 감정이 있지만 화자가 모르는 사이에 녹음 되는 데이터이기 때문에 1번보다 자연적인 데이터이다.  
3) Natural database: 가장 현실적인 데이터이지만 얻기 어려운 데이터이다. 일반 대중 대화, 콜 센터 대화 등이 이에 해당한다.  
Figure3는 이런 데이터베이스를 난이도 별로 정리한 그래프이다.  
![그림5](https://user-images.githubusercontent.com/76423415/122484854-8a1ad300-d010-11eb-9c3a-8379365e4a3a.png)

Table4는 SER분야에서 대중적으로 사용하는 프리소스 음성 데이터베이스를 정리한 도표이다.  
![그림6](https://user-images.githubusercontent.com/76423415/122484867-90a94a80-d010-11eb-890f-e3231fa59ec3.png)  
[역자 의견]  
1,2번의 데이터 베이스가 가장 대중적으로 쓰이는 것이며 source를 확인해보면 연기자들에 의해 수집된 것을 알 수 있다.  

---
### III. NEED OF DEEP LEARNING TECHNIQUES FOR SER : SER에서 딥러닝의 필요성

전통적인 SER에서는 GMMs, HMMs이 사용되었다.  
GMMs(Gaussian Mixture Model): 음향 특징을 설명하기 위해 주로 활용된다.  
HMMs(Hidden Markov Model): 음성 신호의 시간적 변화를 다루기 위해 주로 활용된다.  

딥러닝 방법은 비선형 요소로 구성되어 병렬 구조로 계산되는 모델이다.  
딥러닝 모델로는 Deep Boltzmann Machine (DBM), Recurrent Neural Network (RNN), Recursive Neural Network (RNN), Deep Belief Network (DBN), Convolutional Neural Networks (CNN), Auto Encoder (AE) 등이 있다.  

Figure4는 전통적인 SER 방법과 딥러닝을 활용한 SER을 비교하였다.  
![그림7](https://user-images.githubusercontent.com/76423415/122485493-05c94f80-d012-11eb-9fe8-abbd32ac05cb.png)  
딥러닝 방법을 사용하면 SER의 과정을 압축시킬 수 있다.  

Table5는 IEMOCAP, Emo-DB, SAVEE 데이터셋을 활용하여 여러 딥러닝의 성능을 비교한 도표이다.  
![table5](https://user-images.githubusercontent.com/76423415/122485533-25607800-d012-11eb-85c1-c0d78db03b19.PNG)  
[역자 의견]  
특히 CNN에서 성능이 잘 나온 것을 확인 할 수 있다.  


---
### IV. DEEP LEARNING TECHNIQUES FOR SER : SER을 위한 딥러닝 기술

해당 섹션에서 소개할 딥러닝 기술들은 다음과 같다.
- Deep Boltzmann Machine (DBM)  
- Recurrent Neural Network (RNN)  
- Recursive Neural Network (RNN)  
- Deep Belief Network (DBN)  
- Convolutional Neural Networks (CNN)  
- Auto Encoder (AE)  

A. DEEP BOLTZMANN MACHINE (DBM)  
DBM은 히든 마르코프 모형에서 파생되었으며 다양한 숨겨진 계층으로 구성되어있으며 히든 레이어 안으이 동일한 계층 간의 상호 연결이 있다.  
양방향 추론과 학습을 함으로 모호하고 복잡한 데이터를 잘 학습할 수 있다.  
반지도 학습에 유용하다.  
![그림8](https://user-images.githubusercontent.com/76423415/122485880-ea127900-d012-11eb-9c30-6f4a689413ec.png)


B. RECURRENT NEURAL NETWORK (RNN)  
이전의 출력이 다음의 결과에 영향을 주는 형태의 인공신경망이다.  
유닛간의 연결이 순환적 구조를 가져 시계열 데이터에 적합하다.  
장기의존성 문제를 해결하기 위해 cell state를 추가한 LSTM이 파생되었다.  
![그림10](https://user-images.githubusercontent.com/76423415/122485889-f1d21d80-d012-11eb-9e4f-3ff5481631b2.png)


C. RECURSIVE NEURAL NETWORK (RvNN)  
입력된 데이터를 작은 청크로 나누어 학습하는 신경망이다.  
청크로 나누어 학습하며 데이터 손실이 일어나지만 언어의 계층적인 성질을 닮아 언어 데이터에 적합하다.  
![그림11](https://user-images.githubusercontent.com/76423415/122485902-f696d180-d012-11eb-9146-01db0e867e75.png)


D. DEEP BELIEF NETWORK (DBN)  
RBM(제한된 볼츠만 머신)을 블록처럼 여러 층으로 쌓아 연결한 신경망이다.  
결과값 없이 입력값으로만 학습을 수행한다.(비지도 학습)  
![그림9](https://user-images.githubusercontent.com/76423415/122485904-f8f92b80-d012-11eb-89d4-e67d80fac630.png)


E. CONVOLUTIONAL NEURAL NETWORK (CNN)  
합성곱층(covolutional layer)과 풀링층(pooling layer)로 구성된 신경망이다.  
공간 정보를 유지한 상태로 학습이 가능하기 때문에 이미지, 영상에 주로 쓰인다.  
![그림12](https://user-images.githubusercontent.com/76423415/122485910-fc8cb280-d012-11eb-83ef-9232925a971d.png)


F. AUTO ENCODER (AE)  
네트워크에 여러 제약을 주면서 입력층을 출력층으로 복사하는 신경망아다.  
노이즈 제거나 차원 축소에 적합하다.  
![그림13](https://user-images.githubusercontent.com/76423415/122485915-feef0c80-d012-11eb-955b-89641556a658.png)


---
### V. SUMMARY OF THE LITERATURE, DISCUSSION AND FUTURE DIRECTIONS : 논문 요약 및 향후 방향 논의

딥러닝 기술은 raw data의 특징을 계층적으로 학습하여 여러 이점이 존재한다.  
이 섹션에서는 SER의 관점에서 딥러닝 모델의 구조, 학습 및 정규화 방법론등에 관련된 논문을 요약한다.  
이는 Table7에 해당한다.  
Table7은 논문에서 소개한 딥러닝 모델에 대한 간략한 설명을 정리한 표이다.  
* 모든 테이블은 상단 논문 링크에서 확인할 수 있습니다.*  
* 37개의 논문을 요약한 내용도 본문에서 확인할 수 있습니다.*  
![그림2](https://user-images.githubusercontent.com/76423415/122485971-17f7bd80-d013-11eb-9485-0b4529169c90.png)  
[역자 의견]  
본 논문이 쓰여진 시점에서 가장 최근의 논문은 CNN, LSTM을 엮은 모델을 사용하였는데, 성능이 가장 좋다.  
2018년도의 다른 연구들에도 CNN, RNN, LSTM, DNN 등을 혼합하여 사용하는 것을 확인 할 수 있다.  
기술이 빨리 발전해서 가장 좋은 모델은 최신 논문을 확인해 보아야 할 것 같다.  

---
### VI. CONCLUSION  

SER에 대한 DBM, RNN, DBN, CNN 및 AE와 같은 딥러닝 기술은 최근 몇 년 동안 많은 연구의 대상이 되었다.  
이러한 딥러닝 방법은 anger(분노), boredom(지루함), disgust(혐오), surprise(놀람), fear(두려움), joy(기쁨), happiness(행복), neutral(중립), sadness(슬픔) 등 다양한 자연 감정의 분류를 바탕으로 간략하게 설명된다.  
이러한 방법은 공유 가중치를 활용한 효율성뿐만 아니라 쉬운 모델 훈련을 제공한다.  
그러나 대규모 계층별 내부 아키텍처, 일시적으로 변화하는 input data에 대한 효율성 감소 및 과적합등의 한계가 있다.
이 연구는 현재의 딥 러닝 기법의 성능과 한계를 평가하기 위한 기반을 형성한다.  
또한, 더 나은 SER 시스템을 위한 몇 가지 유망한 방향을 강조하기 위해 작성되었다.
