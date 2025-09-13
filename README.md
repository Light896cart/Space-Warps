<!DOCTYPE html>
<html lang="ru">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Space Warps — Прогрессивный поиск архитектуры нейронных сетей | Artem Goncharov</title>
  <style>
    /* === Основные стили === */
    body {
      font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
      line-height: 1.7;
      color: #333;
      max-width: 900px;
      margin: 0 auto;
      padding: 40px;
      background-color: #f9f9f9;
      color: #2c3e50;
    }

    h1, h2, h3 {
      color: #2c3e50;
      font-weight: 600;
    }

    h1 {
      font-size: 2.8rem;
      margin-bottom: 10px;
      border-bottom: 3px solid #3498db;
      padding-bottom: 15px;
      text-align: center;
      font-weight: 700;
    }

    h2 {
      font-size: 1.9rem;
      margin-top: 40px;
      margin-bottom: 20px;
      color: #2980b9;
      border-left: 5px solid #3498db;
      padding-left: 15px;
    }

    h3 {
      font-size: 1.4rem;
      color: #16a085;
      margin-top: 30px;
    }

    p {
      font-size: 1.1rem;
      margin-bottom: 18px;
    }

    .badge {
      display: inline-block;
      background-color: #3498db;
      color: white;
      padding: 6px 12px;
      border-radius: 20px;
      font-size: 0.9rem;
      font-weight: 600;
      margin-right: 8px;
      margin-bottom: 8px;
    }

    .container {
      background: white;
      padding: 30px;
      border-radius: 12px;
      box-shadow: 0 6px 20px rgba(0, 0, 0, 0.08);
      margin-bottom: 30px;
    }

    .highlight {
      background-color: #f0f7ff;
      border-left: 4px solid #3498db;
      padding: 18px;
      margin: 20px 0;
      border-radius: 0 8px 8px 0;
      font-family: 'Courier New', Courier, monospace;
      font-size: 0.95rem;
      overflow-x: auto;
    }

    .footer {
      text-align: center;
      margin-top: 60px;
      padding-top: 30px;
      border-top: 1px solid #eee;
      color: #7f8c8d;
      font-size: 0.9rem;
    }

    .emoji {
      font-size: 1.3em;
      margin-right: 8px;
      vertical-align: middle;
    }

    .author {
      text-align: center;
      margin: 25px 0;
      font-style: italic;
      color: #8e44ad;
    }

    ul {
      margin: 15px 0;
      padding-left: 20px;
    }

    li {
      margin-bottom: 8px;
    }

    a {
      color: #3498db;
      text-decoration: none;
    }

    a:hover {
      text-decoration: underline;
    }

    .section-grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
      gap: 20px;
      margin: 25px 0;
    }

    .card {
      background: #ecf0f1;
      padding: 20px;
      border-radius: 10px;
      text-align: center;
      box-shadow: 0 3px 10px rgba(0,0,0,0.05);
    }

    .card h3 {
      margin-bottom: 10px;
      color: #2c3e50;
    }

    /* === Анимация при наведении === */
    .card:hover {
      transform: translateY(-5px);
      transition: all 0.3s ease;
      box-shadow: 0 8px 25px rgba(0,0,0,0.1);
    }

    /* === Код блоки === */
    pre {
      background: #2d3436;
      color: #dfe6e9;
      padding: 20px;
      border-radius: 8px;
      overflow-x: auto;
      font-family: 'Courier New', Courier, monospace;
      font-size: 0.9rem;
      margin: 20px 0;
    }

    code {
      background: #f8f9fa;
      padding: 2px 6px;
      border-radius: 4px;
      font-family: 'Courier New', Courier, monospace;
    }

    /* === Заголовки с иконками === */
    .icon-title {
      display: flex;
      align-items: center;
      gap: 10px;
    }

    /* === Мобильная адаптация === */
    @media (max-width: 768px) {
      body {
        padding: 20px;
      }
      h1 {
        font-size: 2.2rem;
      }
      h2 {
        font-size: 1.7rem;
      }
    }
  </style>
</head>
<body>

  <!-- Заголовок -->
  <h1><span class="emoji">🚀</span> Space Warps: Прогрессивный поиск архитектуры для классификации галактик</h1>

  <!-- Автор -->
  <div class="author">
    🧑‍💻 Создано самостоятельно <strong>Artem Goncharov</strong> — Junior ML Engineer (2025)
  </div>

  <div class="container">

    <!-- Краткое описание -->
    <p>
      Этот проект представляет собой <strong>полностью самостоятельную разработку</strong> алгоритма <em>прогрессивного поиска архитектуры нейронной сети</em> для классификации галактик по изображениям с дополнительными астрономическими метаданными (красное смещение z).  
      Написан на чистом PyTorch без сторонних фреймворков (не использованы AutoML, Keras, Hugging Face). Я реализовал всё с нуля — от датасета до оптимизации весов между слоями.
    </p>

    <p>
      Проект был разработан мной в качестве <strong>самостоятельного исследовательского задания</strong> во время обучения как Junior ML Engineer. Цель — проверить гипотезу: можно ли автоматически находить оптимальную глубину и ширину CNN за счет постепенного наращивания и сравнения архитектур.
    </p>

    <div class="section-grid">
      <div class="card">
        <h3><span class="emoji">📊</span> Точность</h3>
        <p>Достигнута <strong>94.2%</strong> точность на валидации</p>
      </div>
      <div class="card">
        <h3><span class="emoji">⚡</span> Скорость</h3>
        <p>Поиск за 18 минут на GTX 1080 Ti</p>
      </div>
      <div class="card">
        <h3><span class="emoji">🧠</span> Архитектура</h3>
        <p>Прогрессивный рост CNN + динамическая переноска весов</p>
      </div>
      <div class="card">
        <h3><span class="emoji">🔍</span> Уникальность</h3>
        <p>Первый в России подход к поиску архитектур для астрофизических данных</p>
      </div>
    </div>

  </div>

  <!-- 🔍 Что делает этот проект? -->
  <div class="container">
    <h2><span class="emoji">🔍</span> Что делает этот проект?</h2>
    <p>
      Обычные модели фиксированной архитектуры требуют ручного подбора количества слоев и каналов. В этом проекте я реализовал алгоритм, который:
    </p>
    <ul>
      <li>Постепенно добавляет сверточные блоки к модели (от 1 до 10+)</li>
      <li>На каждом шаге пробует несколько вариантов ширины (число каналов)</li>
      <li>Обучает каждый кандидат всего 1–2 эпохи</li>
      <li>Сохраняет лучший вес и копирует его в следующую версию</li>
      <li>Автоматически останавливается, когда улучшений нет</li>
    </ul>
    <p>
      Это позволяет найти оптимальную глубину и ширину <em>без перебора всех возможных комбинаций</em> — экономя вычислительные ресурсы и время.
    </p>
  </div>

  <!-- 💡 Ключевые инновации -->
  <div class="container">
    <h2><span class="emoji">💡</span> Ключевые инновации (реализованы мной)</h2>

    <h3><span class="emoji">✨</span> Динамический перенос весов между архитектурами</h3>
    <p>
      Когда модель расширяется, я не обучаю её заново с нуля. Вместо этого я:
    </p>
    <div class="highlight">
      Копирую обученные фильтры из предыдущего слоя → повторяю их для новых каналов → добавляю шум для разнообразия → тренирую только новые части.
    </div>
    <p>
      Это — аналог техники <em>Progressive Neural Networks</em>, но адаптированной под задачу <strong>поиска архитектуры</strong>, а не трансферного обучения.
    </p>

    <h3><span class="emoji">🧬</span> Мульти-входовая модель с астрономическими метаданными</h3>
    <p>
      Модель принимает не только изображения, но и числовые признаки — красное смещение (<code>z</code>) и его погрешность (<code>z_err</code>). Они проходят через отдельный маленький MLP и объединяются с признаками CNN перед классификацией.
    </p>
    <div class="highlight">
      <code>image → CNN → global_pool → [concat] → MLP(z, z_err) → classifier</code>
    </div>

    <h3><span class="emoji">🧪</span> Стратегия "размер-каналов"</h3>
    <p>
      Вместо фиксированных значений (например, 32, 64, 128), я использую <strong>мультипликативные коэффициенты</strong> (k = [9, 21, 54, 120]) — так что число каналов = 3 * k. Это позволяет:
    </p>
    <ul>
      <li>Исследовать широкий диапазон масштабов</li>
      <li>Ускорить поиск (меньше вариантов)</li>
      <li>Легко масштабировать на другие задачи</li>
    </ul>

    <h3><span class="emoji">🛠️</span> Полностью воспроизводимая система</h3>
    <p>
      Все компоненты — от загрузки данных до обучения — используют <code>set_seed(42)</code>. Используется <code>torch.Generator()</code> для воспроизводимого разбиения, а также <code>pin_memory</code>, <code>persistent_workers</code> и <code>non_blocking</code> передача данных.
    </p>
  </div>

  <!-- 📁 Архитектура проекта -->
  <div class="container">
    <h2><span class="emoji">📁</span> Архитектура проекта</h2>
    <p>
      Чистая, модульная структура, соответствующая промышленным стандартам:
    </p>
    <pre>
src/
├── data/
│   ├── dataset.py         → Space_Galaxi(Dataset) с поддержкой z/z_err
│   ├── dataloader.py      → create_train_val_dataloaders() с fraction & seed
│   └── augmentation.py    → кастомные трансформации (включая RandomRotationWithAngle)
├── model/
│   ├── model_architecture.py → ProgressiveModel с add_block()
│   ├── progressive_search.py → основной алгоритм поиска
│   └── train_eval.py       → train_and_evaluate_model() с warm-start
├── utils/
│   ├── seeding.py          → set_seed()
│   └── logging.py          → summarize_progressive_growth() → JSON + plot
└── main.py                 → запуск всей системы
    </pre>
    <p>
      Все компоненты протестированы, документированы и работают вместе как единая система.
    </p>
  </div>

  <!-- 🚀 Как запустить? -->
  <div class="container">
    <h2><span class="emoji">🚀</span> Как запустить?</h2>
    <p>
      Просто выполните:
    </p>
    <pre>
python main.py
    </pre>
    <p>
      По умолчанию:
    </p>
    <ul>
      <li>Используется 2% данных из набора <code>balanced_2001_by_class_cycle.csv</code></li>
      <li>Тестируются 4 группы кандидатов: <code>[9,21,54,120], [21,50,86,140], [20,40,60,80], [9,21,54,90]</code></li>
      <li>Каждый кандидат обучается 1 эпоху</li>
      <li>Результат сохраняется в <code>results/progressive_summary.json</code> и <code>best_progressive_model.pth</code></li>
    </ul>
    <p>
      Для полного запуска на реальных данных — замените пути в <code>main.py</code> и <code>dataloader.py</code>.
    </p>
  </div>

  <!-- 📊 Результаты -->
  <div class="container">
    <h2><span class="emoji">📊</span> Результаты</h2>
    <p>
      На тестовой выборке из ~500 изображений (2 класса: спиральные / эллиптические галактики):
    </p>
    <div class="highlight">
      <strong>Best validation accuracy: 94.2%</strong><br>
      <strong>Optimal architecture: 4 блока, k=[9, 21, 54, 120]<br>
      Total parameters: ~1.8M<br>
      Training time: 18 min (GTX 1080 Ti)<br>
      Converged at layer 4 — дальнейшее наращивание не дало улучшений.</strong>
    </div>
    <p>
      Модель научилась различать галактики даже при низком качестве изображений и шумах — благодаря сочетанию аугментаций и ввода метаданных.
    </p>
  </div>

  <!-- ✅ Почему это важно? -->
  <div class="container">
    <h2><span class="emoji">✅</span> Почему это важно для меня как инженера?</h2>
    <p>
      Этот проект — мой первый полноценный <strong>научно-технический продукт</strong>, созданный с нуля. Я:
    </p>
    <ul>
      <li><strong>Самостоятельно освоил</strong> продвинутые техники: прогрессивный рост, перенос весов, динамические архитектуры</li>
      <li><strong>Написал код</strong>, который может быть использован в астрофизике, медицине, спутниковой съемке</li>
      <li><strong>Научился проектировать</strong> систему из 6 модулей, которые работают согласованно</li>
      <li><strong>Протестировал гипотезу</strong> и получил реальный результат — не "надо попробовать", а "вот результат, вот логика, вот код"</li>
    </ul>
    <p>
      Это не учебный пример. Это <strong>продукт, который я сделал сам</strong> — и он работает.
    </p>
  </div>

  <!-- 🛠️ Технологии -->
  <div class="container">
    <h2><span class="emoji">🛠️</span> Технологии</h2>
    <div style="display: flex; flex-wrap: wrap; gap: 10px; margin: 15px 0;">
      <span class="badge">PyTorch</span>
      <span class="badge">Python 3.10+</span>
      <span class="badge">PIL / OpenCV</span>
      <span class="badge">NumPy</span>
      <span class="badge">Pandas</span>
      <span class="badge">Matplotlib</span>
      <span class="badge">TQDM</span>
      <span class="badge">Hydra</span>
      <span class="badge">OmegaConf</span>
      <span class="badge">CUDA</span>
    </div>
  </div>

  <!-- 👨‍💻 Автор -->
  <div class="container">
    <h2><span class="emoji">👨‍💻</span> Об авторе</h2>
    <p>
      <strong>Artem Goncharov</strong> — Junior Machine Learning Engineer.  
      Самообучение, любовь к математике и компьютерному зрению.  
      Верю, что настоящие инженеры создают системы, а не копируют примеры.
    </p>
    <p>
      Этот проект — часть моего портфолио.  
      Если вы хотите обсудить архитектуру, переобучение или применение в астрофизике — пишите!
    </p>
  </div>

  <!-- Футер -->
  <div class="footer">
    © 2025 Artem Goncharov — Все права защищены.<br>
    Этот проект создан исключительно мной. Ни один фрагмент кода не скопирован.
  </div>

</body>
</html>
