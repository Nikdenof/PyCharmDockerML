# ML Dev Project Template for PyCharm Professional 

A detailed description of the entire project structure can be found in this [Medium article](https://medium.com/@nikdenof/2024-extensive-guide-building-ml-applications-with-docker-compose-and-pycharm-professional-695821e5243e).

---
## Jupyter Notebooks
To launch Jupyter Notebook server use the following command:
```bash
docker-compose -f docker-compose.dev.yml up -d - build
```


---

## Tests
To run test using development container use the following command:
```bash
docker-compose -f docker-compose.dev.yml run dev_container  pytest /app/tests/test_model.py
```

---



