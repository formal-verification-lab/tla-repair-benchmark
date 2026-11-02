# GigaChat API

Клиент реализует общий `LLMClient` и использует персональный scope
`GIGACHAT_API_PERS`. Для вызовов достаточно стандартной библиотеки Python.
Покупка пакетов, переход на платный тариф и замена модели автоматически не выполняются.
Фактические квоты следует проверить в своём аккаунте перед экспериментами.

## Авторизация и TLS

Для получения access token нужен Base64-ключ авторизации из проекта GigaChat API.
Передайте его через `GIGACHAT_CREDENTIALS` или отдельный файл с правами `0600`
через `--credentials-file`. Не включайте префикс `Basic` в значение ключа.
Не храните ключ в сообщениях, Git или параметрах командной строки.

На подготовленной машине файл расположен в
`~/.config/tla-repair/gigachat-authorization.key`. При перевыпуске ключа можно
без отображения в терминале заменить его следующей командой:

```sh
python3 -c 'import getpass, os; from pathlib import Path; p=Path.home()/".config/tla-repair/gigachat-authorization.key"; p.parent.mkdir(parents=True, exist_ok=True); key=getpass.getpass("GigaChat authorization key: "); fd=os.open(p, os.O_WRONLY|os.O_CREAT|os.O_TRUNC, 0o600); os.fchmod(fd, 0o600); os.write(fd, (key+"\n").encode()); os.close(fd)'
```

Для TLS GigaChat требуются сертификаты НУЦ Минцифры. Их можно скачать с
[портала Госуслуг](https://www.gosuslugi.ru/crt), объединить корневой и выпускающий
PEM-сертификаты в один файл и передать через `--ca-bundle`.
На подготовленной машине это `~/.config/tla-repair/gigachat-ca.pem`.
Сертификаты добавляются только в SSLContext клиента; системное хранилище
не изменяется, проверка TLS не отключается. В артефактах фиксируется SHA256 bundle.

## Проверка доступа

Из корня репозитория после установки `pip install -e .`:

```sh
python -m benchmark gigachat-access \
  --credentials-file "$HOME/.config/tla-repair/gigachat-authorization.key" \
  --ca-bundle "$HOME/.config/tla-repair/gigachat-ca.pem" \
  --output results/gigachat-access
```

Команда получает список моделей и остаток токенов. Вызов генерации не выполняется.
Наличие модели в списке не является доказательством бесплатного доступа:
нужно также проверить квоты и условия Freemium своего аккаунта.

## Генерация

`--messages` принимает JSON-массив объектов `role`/`content`, как клиент Ollama.
История передаётся явно, скрытых чатов нет. Пример вызова с подготовленным файлом:

```sh
python -m benchmark gigachat-chat \
  --credentials-file "$HOME/.config/tla-repair/gigachat-authorization.key" \
  --ca-bundle "$HOME/.config/tla-repair/gigachat-ca.pem" \
  --model GigaChat-3-Ultra \
  --expected-model GigaChat-3-Ultra:32.9.16.9 \
  --messages messages.json \
  --max-output 2048 --temperature 0.1 \
  --output results/gigachat-pilot
```

Версия `GigaChat-3-Ultra:32.9.16.9` наблюдалась в локальном пилоте 2 ноября 2026.
Это не гарантия неизменности сервиса. Перед основными прогонами нужно проверить
текущую версию и зафиксировать её. `--expected-model` сравнивает фактическую версию
из ответа и останавливает прогон при несовпадении; расход уже выполненного
запроса сохраняется. Клиент не переключает модель и не повторяет генерацию.

## Артефакты и ошибки

Каждый вызов создаёт отдельный `calls/<call-id>/` с запросом, метаданными,
исходным ответом, результатом или ошибкой. OAuth-ответы, access token и ключ
авторизации не записываются. Access token обновляется перед вызовом после
истечения срока, без повторного обращения к модели. Redirect запрещён.
HTTP-ошибки сохраняют статус и `Retry-After`; повтор и ожидание должен выполнять
runner с отдельным учётом технических сбоев. Тело HTTP-ошибки не сохраняется,
поскольку сервер может отразить в нём секреты. Ответ генерации сохраняется
побайтно, кроме защитного удаления секретов при их неожиданном отражении.

`Usage.input_tokens` включает полный вход: `prompt_tokens` плюс
`precached_prompt_tokens`. У GigaChat `prompt_tokens` уже исключает кэш;
`total_tokens` означает тарифицируемый объём. Эти исходные поля сохраняются
отдельно в ответе. Неизвестные счётчики остаются `null`.
Окончание генерации по лимиту не означает успех SANY/TLC.

## Наблюдение пилота

В аккаунте подтверждены квоты, соответствующие Freemium, включая 50 млн токенов
Ultra. На синтетической Mutex2 модель прошла SANY на попытке 2 и TLC на попытке 5;
TLC обнаружил ожидаемые три состояния. Передавались исходное описание, неизменный
`.cfg` и полные логи инструментов, без эталона и ручных исправлений.
Изначальный предел три попытки был расширен до десяти после просмотра результата:
это диагностический пилот, не основной эксперимент. Обе сводки сохранены раздельно
в локальном `results/gigachat-access-20261102T114301Z-8a8d31/`.
Исходные артефакты первого пилота сохранены; уточнённый учёт кэшированных токенов
вынесен в `token-accounting-audit.json`.
Одна синтетическая задача не подтверждает качество на четырёх задачах курсовой.

Источники: [авторизация](https://developers.sber.ru/docs/ru/gigachat/api/reference/rest/gigachat-api),
[сертификаты](https://developers.sber.ru/docs/ru/gigachat/certificates),
[Freemium](https://developers.sber.ru/docs/ru/gigachat/tariffs/individual-tariffs),
[счётчики токенов](https://developers.sber.ru/docs/ru/gigachat/guides/counting-tokens).
