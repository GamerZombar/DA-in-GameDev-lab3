# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #2 выполнил(а):
- Романов Вадим Юрьевич
- РИ210950
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | * | 20 |
| Задание 3 | * | 20 |

знак "*" - задание выполнено; знак "#" - задание не выполнено;

Работу проверили:
- к.т.н., доцент Денисов Д.В.
- к.э.н., доцент Панов М.А.
- ст. преп., Фадеев В.О.

[![N|Solid](https://cldup.com/dTxpPi9lDf.thumb.png)](https://nodesource.com/products/nsolid)

[![Build Status](https://travis-ci.org/joemccann/dillinger.svg?branch=master)](https://travis-ci.org/joemccann/dillinger)

Структура отчета

- Данные о работе: название работы, фио, группа, выполненные задания.
- Цель работы.
- Задание 1.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 2.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Задание 3.
- Код реализации выполнения задания. Визуализация результатов выполнения (если применимо).
- Выводы.
- ✨Magic ✨

## Цель работы
Разработать систему машинного обучения

## Задание 1
### Реализовать запись в Google-таблицу набора данных, полученных с помощью линейной регрессии из лабораторной работы № 1
Ход работы:
- Создал новый 3D проект в Unity, с помощью Package Manager'а добавил пакет MLAgent и extensions для него.
- Добавил на сцену плоскость, куб и сферу, изменил их свойства и добавил материалы, для наглядности. Для сферы был добавлен Rigidbody и написан скрипт

```cs
using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    public Transform Target;
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        Target.localPosition = new Vector3(Random.value * 8 - 4, 0.5f, Random.value * 8 - 4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(Target.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        float distanceToTarget = Vector3.Distance(this.transform.localPosition, Target.localPosition);

        if (distanceToTarget < 1.42f)
        {
            SetReward(1.0f);
            EndEpisode();
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
```

- Открыл Anaconda Promt, куда вписал следющие команды, для создания и активации MLAgent: 

```py
conda create -n MLAgent python=3.6
```

```py
conda activate MLAgent
```

```py
pip install mlagents
pip install torch~=1.7.1 -f https://download.pytorch.org/whl/torch_stable.html
```

####После установки всех пакетов и создания MLAgent'а, перешел в директорию с Unity проектом, где, благодаря заготовленным конфигурационным данным в `.yaml` файле, запустил обучаться ML-агент. 

![92](https://user-images.githubusercontent.com/58142149/200951860-aefe1770-fbc8-4d79-8f63-6ebf0d0a7830.png)

- После тестового прогона, на одном экземпляре, увеличил количество копий, начиная с 9, заканчивая 45 (после этого юнька начала жутко лагать, поэтому остановился на этом числе...)

![Screenshot_8](https://user-images.githubusercontent.com/58142149/200951931-3eccfcc1-0e70-4b3d-bbfb-a01e5de12ff9.png)

- По итогу, остановился на 40000 шагов, на выходе получились следующие данные: 
`[INFO] RollerBall. Step: 40000. Time Elapsed: 319.171 s. Mean Reward: 0.993. Std of Reward: 0.085. Training.`

Проверил результат обучения MLAgent'а на одном экземпляре... результат, если честно, удивил) 

____

## Задание 2
### Подробно описать каждую строку файла конфигурации нейронной сети. Самостоятельно найти информацию о компонентах Decision Requester, Behavior Parameters, добавленных сфере

```yaml

behaviors:
  RollerBall:  # ID агента
    trainer_type: ppo  # Режим обучения - Proximal Policy Optimization
    hyperparameters:
      batch_size: 10  # Кол-во примеров, в одной итерации
      buffer_size: 100  # Кол-во опыта, которое необходимо набрать перед обновлением модели
      learning_rate: 3.0e-4  # Начальная скорость обучения
      beta: 5.0e-4  # Сила регуляции энтропии, увеличивает случайность действий
      epsilon: 0.2  # Порог расхождений между старой и новой политиками при обновлении
      lambd: 0.99  # Параметр регуляции - насколько сильно агент полагается на текущий value estimate
      num_epoch: 3  # Кол-во проходов через буфер опыта, при выполнении оптимизации
      learning_rate_schedule: linear  # Определяет, как скорость обучения изменяется с течением времени, linear - линейно уменьшает скорость
    network_settings:  
      normalize: false  # Отключаем нормализацию входных данных
      hidden_units: 128  # Кол-во нейронов в скрытых слоях сети
      num_layers: 2  # Кол-во скрытых слоёв в сети
    reward_signals:
      extrinsic:
        gamma: 0.99  # Коэффициент "скидки" для будущих вознаграждений
        strength: 1.0  # Коэффициент, на который умножается вознаграждение
    max_steps: 500000  # Общее кол-во шагов, которые должны быть выполнены в среде, до завершения обучения
    time_horizon: 64  # Сколько опыта нужно собрать для каждого агента, прежде чем добавлять его в буфер
    summary_freq: 10000  # Кол-во опыта, который необходимо собрать перед созданием и отображением статистики

```

`Decision Requester` - запрашивает решение через регулярные промежутки времени

`Behavior Parameters` - определяет принятие объектом решений, в него указывается какой тип поведения будет использоваться: уже обученная модель или удалённый процесс обучения

____

## Задание 3
### Доработайте сцену и обучите ML-Agent таким образом, чтобы шар перемещался между двумя кубами разного цвета. Кубы должны, как и впервом задании, случайно изменять кооринаты на плоскости

- Добавил на сцену ещё один куб
- Изменил скрипт RollerAgent, под текущую задачу, добавив один флаг и дополнительный прощет дистанции до второго куба

```cs
    using System.Collections;
using System.Collections.Generic;
using UnityEngine;
using Unity.MLAgents;
using Unity.MLAgents.Sensors;
using Unity.MLAgents.Actuators;

public class RollerAgent : Agent
{
    Rigidbody rBody;
    // Start is called before the first frame update
    void Start()
    {
        rBody = GetComponent<Rigidbody>();
    }

    private bool isTouchFirstTarget; // added

    public Transform FirstTarget;  // added (From Target --> FirstTarget)
    public Transform SecondTarget; // added
    public override void OnEpisodeBegin()
    {
        if (this.transform.localPosition.y < 0)
        {
            this.rBody.angularVelocity = Vector3.zero;
            this.rBody.velocity = Vector3.zero;
            this.transform.localPosition = new Vector3(0, 0.5f, 0);
        }

        FirstTarget.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);

        SecondTarget.localPosition = new Vector3(Random.value * 8-4, 0.5f, Random.value * 8-4);
    }
    public override void CollectObservations(VectorSensor sensor)
    {
        sensor.AddObservation(FirstTarget.localPosition);
        sensor.AddObservation(SecondTarget.localPosition);
        sensor.AddObservation(this.transform.localPosition);
        sensor.AddObservation(rBody.velocity.x);
        sensor.AddObservation(rBody.velocity.z);
    }
    public float forceMultiplier = 10;
    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        Vector3 controlSignal = Vector3.zero;
        controlSignal.x = actionBuffers.ContinuousActions[0];
        controlSignal.z = actionBuffers.ContinuousActions[1];
        rBody.AddForce(controlSignal * forceMultiplier);

        //edited code
        float distanceToFirstTarget = Vector3.Distance(this.transform.localPosition, FirstTarget.localPosition);

        float distanceToSecondTarget = Vector3.Distance(this.transform.localPosition, SecondTarget.localPosition);


        if(distanceToFirstTarget < 1.42f && !isTouchFirstTarget)
        {
            SetReward(1.0f);
            EndEpisode();
            isTouchFirstTarget = !isTouchFirstTarget;
        }
        else if (distanceToSecondTarget < 1.42f && isTouchFirstTarget){
            SetReward(1.0f);
            EndEpisode();
            isTouchFirstTarget = !isTouchFirstTarget;
        }
        else if (this.transform.localPosition.y < 0)
        {
            EndEpisode();
        }
    }
}
    
```

- Обучаемся...
![Screenshot_9](https://user-images.githubusercontent.com/58142149/201138359-332f6562-7b8b-4fab-88b6-ddd585f0b130.png)

- На 150000 решил остановиться)
![Screenshot_11](https://user-images.githubusercontent.com/58142149/201138221-70f569e4-28a4-48a9-a9b8-83ab5add965e.png)


И вот результат:

![Анимация](https://user-images.githubusercontent.com/58142149/201137823-466e53e5-becb-41c5-8d2a-0ebc205e0c34.gif)


____

## Выводы

Игровой баланс - это баланс характеристик объектов игрового взаимодействия. Описания баланса зависит от типа игры, и от объектов которые необходимо балансировать. Абсолютного баланса найти невозможно, так как это не точный набор данных, который можно посчитать, а представление, которое формируется как и математическими вычислениями, так и количеством проанализированных игровых сессий.

Системы машинного обучения, или же нейросети, могут быть использованы для корретировки игрового баланса, к примеру, анализом игровых сессий игроков и генерированием оптимальных уровней или противников. В singleplay играх машинное обучение может изменять сложность игровых испытаний и поведение вражеских юнитов, в зависимости от поведения игрока и количества попыток, которые были им затрачены. Ну и, как упомянул выше, каждый из сценариев изпользования машинного обучения в балансировке игры будет зависить от конкретной игры и объектов которые необходимо сбалансировать.


## Powered by

**BigDigital Team: Denisov | Fadeev | Panov**

