# API

在仓库根目录执行：

```sh
python -m pip install -r api/requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

接口文档：`http://127.0.0.1:8000/docs`。

| 接口 | 功能 |
| --- | --- |
| `GET /api/health` | 服务状态与目录版本 |
| `GET /api/catalog` | 完整目录、坐标说明与版本，供客户端一次加载 |
| `GET /api/landslides` | 目录查询，支持 `q`、`type`、`page`、`pageSize` |
| `GET /api/landslides/{id}` | 记录详情 |
| `GET /api/stats` | 类型统计，支持与列表相同的 `q`、`type` |
| `GET /api/capabilities` | 分类、检测服务可用状态 |
| `POST /api/landslide/classify` | 分类入口，当前返回 `503 MODEL_NOT_CONFIGURED` |

默认读取 `examples/catalog.demo.json`，所有记录均为合成演示数据。设置环境变量 `MARS_CATALOG_PATH` 可读取同结构的本地 JSON；目录在服务启动时校验，修改后需重启。经纬度采用目录声明的火星坐标，原值返回。不存在的记录返回 404，非法查询参数返回 422。

`MARS_CORS_ORIGINS` 可设置允许访问的前端来源，多个来源用逗号分隔。默认允许本机 Vite 与 Capacitor 页面。手机浏览器联调时可将服务绑定到 `0.0.0.0`，使用电脑局域网地址，并添加实际页面来源；Android封装APP默认使用HTTPS服务，不默认放行明文请求。

测试：

```sh
python -m pip install -r api/requirements-dev.txt
python -m pytest api/tests -q
```
