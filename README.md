# АНАЛИЗ ДАННЫХ И ИСКУССТВЕННЫЙ ИНТЕЛЛЕКТ [in GameDev]
Отчет по лабораторной работе #2 выполнил(а):
- Романов Вадим Юрьевич
- РИ210950
Отметка о выполнении заданий (заполняется студентом):

| Задание | Выполнение | Баллы |
| ------ | ------ | ------ |
| Задание 1 | * | 60 |
| Задание 2 | # | 20 |
| Задание 3 | # | 20 |

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

____

## Задание 3

____

## Выводы

