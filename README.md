# hallu-traj

`hallu-traj` — исследовательская Python-библиотека для оценки риска галлюцинаций в ответах языковых моделей. Она прослеживает, как меняется скрытое состояние каждого токена от слоя к слою, и вычисляет признаки траектории, включая скорость изменения, кривизну и локальное растяжение. Обученный классификатор выдаёт оценку риска от 0 до 1 для ответа целиком и для отдельных токенов.

В интерфейсе Gradio можно сгенерировать ответ модели или ввести свой, увидеть выделенные фрагменты и сравнить профиль выбранного токена со средним профилем ответа. Оценка служит сигналом для дополнительной проверки фактов. Разметка при обучении задаётся на уровне ответа, поэтому подсветка токенов показывает, где детектор обнаружил высокий риск.

## Установка

Понадобится Python 3.10 или новее. Откройте терминал в папке проекта, где лежит `pyproject.toml`. Если вы находитесь в `hallu_traj_library`, сначала выполните `cd hallu_traj`.

**Windows (PowerShell)**

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -e ".[ui]"
```

**Linux и macOS**

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[ui]"
```

Для использования только Python API достаточно `python -m pip install -e .`. Дополнение `ui` устанавливает Gradio и pandas, `llm` — зависимости для работы с моделями Hugging Face, `eval` — загрузчик датасетов для TruthfulQA, `dev` — pytest.

## Быстрый запуск без языковой модели

```bash
python -m hallu_traj.app --mock
```

Откройте в браузере `http://127.0.0.1:7860`. Введите `What is the capital of France?` и нажмите **Проверить**. Чтобы увидеть подсветку на синтетическом примере, введите `halluc sample question`. В этом режиме приложение обучает детектор на синтетических траекториях прямо при запуске. Для другого порта добавьте `--port 7861`. Чтобы остановить сервер, нажмите `Ctrl+C` в терминале.

## Запуск с моделью Hugging Face

Для этого режима нужна модель, которую Transformers может загрузить через `AutoModelForCausalLM`, и детектор, обученный на её скрытых состояниях. При первой загрузке модели и датасета требуется интернет, если соответствующих файлов ещё нет в локальном кэше.

Установите зависимости и запустите обучение детектора на размеченных ответах TruthfulQA. В PowerShell введите идентификатор модели в формате `владелец/модель` или путь к её локальной папке.

```powershell
python -m pip install -e ".[llm,ui,eval]"
$model = Read-Host 'Идентификатор или путь к модели'
python -m hallu_traj.experiments.truthfulqa --model "$model" --protocol reference --n 100 --k-samples 0 --save-detector --out runs/truthfulqa
python -m hallu_traj.app --model "$model" --detector runs/truthfulqa/detector.joblib
```

В Bash для ввода модели используйте `read -r -p 'Идентификатор или путь к модели: ' model`. Обе команды запуска с `"$model"` остаются такими же. Если детектор для вашей модели уже обучен, укажите путь к его `.joblib` в `--detector`. Для запуска интерфейса с готовым детектором достаточно дополнений `llm` и `ui`.

По умолчанию библиотека выбирает доступное устройство в порядке CUDA, MPS, CPU. Обе команды принимают `--device` и `--dtype`. Для запуска на CPU можно добавить `--device cpu --dtype float32`.

## Эксперименты

Скрипт `hallu_traj.experiments.truthfulqa` сравнивает траекторные признаки с базовыми методами на основе энтропии, отрицательной логарифмической вероятности токенов и согласованности нескольких ответов. Кросс-валидация группирует ответы по вопросам. Отчёт содержит AUROC, AUPRC, FPR при TPR 95% и бутстреп-интервалы для AUROC.

- `--protocol reference` оценивает размеченные эталонные ответы. `--protocol generated` оценивает ответы, сгенерированные моделью, и присваивает метки по сходству с эталонами. Ответы с неопределённой меткой пропускаются.
- `--k-samples 0` отключает измерение согласованности и дополнительную генерацию. Чтобы включить её, задайте, например, `--k-samples 5`.
- В папке `--out` сохраняются `results.json` с метриками, `scores.csv` с оценками по ответам и `cache.pkl` для возобновления сбора данных с теми же параметрами. Флаг `--save-detector` также сохраняет `detector.joblib`.

Для проверки конвейера без загрузки модели и датасета запустите синтетический эксперимент. В примере ограничен параллелизм OpenMP до одного потока.

```powershell
$env:OMP_NUM_THREADS = '1'
python -m hallu_traj.experiments.truthfulqa --model mock --protocol reference --n 30 --k-samples 0 --boot 100 --out runs/mock
```

В Bash используйте `export OMP_NUM_THREADS=1` перед командой. Метрики этого запуска описывают работу на синтетических данных.

## Использование из Python

Минимальный пример обучает детектор на синтетических ответах и выводит общую оценку риска и выделенные фрагменты.

```python
from hallu_traj import HallucinationService, TrajectoryDetector, states_to_features
from hallu_traj.mock import MockExtractor

extractor = MockExtractor()
examples = [(f"sample question {i}", 0) for i in range(6)]
examples += [(f"halluc sample question {i}", 1) for i in range(6)]

features = [
    states_to_features(extractor.analyze(question).states)
    for question, _ in examples
]
labels = [label for _, label in examples]
detector = TrajectoryDetector().fit(features, labels)

service = HallucinationService(extractor, detector)
result = service.analyze("halluc new question")

print(f"Риск для ответа: {result.answer_score:.2f}")
print([span.text.strip() for span in result.spans])
```

`service.analyze(question, answer)` принимает готовый ответ. В результате также доступны `token_scores` с оценками токенов и `profile("speed", token=0)` для получения номеров слоёв и значений скорости первого токена.

## Тесты

```bash
python -m pip install -e ".[dev,ui]"
python -m pytest
```

Лицензия [MIT](LICENSE).
