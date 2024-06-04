# Multilayer Perceptron (MLP)

Trabalho 1 de inteligência artificial sobre redes neurais (Multilayer Perceptron).

## Como executar

* Ativar o ambiente virtual que contém as dependências
* Executar o arquivo main.py

> source venv/bin/activate  
> python3 main.ṕy

As dependências necessárias para execução estão especificadas no arquivo **requirements**

## Execução

A MLP pode ser executada realizando um treinamento do zero a partir da execução da função **run_cross_validation** ou informando um arquivo de pesos a partir da função **run_with_carry_weights**

### Execução com Cross Validation

    def main():
        data_mlp = create_data_mlp()
        params = Parameters(0.5, 1, 120, 150, 26, 50)
        run_cross_validation(data_mlp, params, 13)


### Execução com carregamento de pesos

    def main():
        data_mlp = create_data_mlp()
        params = Parameters(0.5, 1, 120, 150, 26, 50)
        file_path = "mlp/logs/cross_13/cross_validation_0_24-05-22-23-53-36/mlp_weights_24-05-22-23-53-36.log"
        run_with_carry_weights(data_mlp, params, file_path, 899)