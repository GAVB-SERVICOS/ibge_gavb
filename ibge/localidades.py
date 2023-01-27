"""
API do IBGE
https://servicodados.ibge.gov.br/api/docs/

"""

import requests
import json
import urllib3
import ssl


headers = {
    "Content-Type": "application/json;charset=UTF-8",
    "User-Agent": "ibge_gavb.py - https://github.com/GAVB-SERVICOS/ibge_gavb",
    "Accept": "application/json, text/plain, */*",
    "Accept-Encoding": "gzip, deflate, br",
    "Accept-Language": "pt-BR,pt;q=0.9,en-US;q=0.8,en;q=0.7",
    "Connection": "keep-alive",
}


class CustomHttpAdapter(requests.adapters.HTTPAdapter):
    """
    Essa classe é responsável por transportar o adapter que permite usar
    o ssl_context padrão evitando o erro unsafe legacy renegotiation disabled.
    """

    def __init__(self, **kwargs):
        """
        Inicia o ssl_context.

        :param ssl_context: Configuração do ssl
        :type: Class, default=None
        :param Kwargs: Parâmetro extra
        :type: any

        """
        ctx = ssl.create_default_context(ssl.Purpose.SERVER_AUTH)
        ctx.options |= 2186412244  # 0x4 OP_LEGACY_SERVER_CONNECT
        self.ssl_context = ctx
        super().__init__(**kwargs)

    def init_poolmanager(self, connections: int, maxsize: int, block: bool = False):
        """
        Configura a PoolManager.

        :param conections: Número de conexões pool para armazenar
                           em cache antes de descartar o mínimo.
        :type: int
        :param maxsize: Método da classe PoolManager.
        :type: int
        :param block: Método da classe PoolManager.
        :type: bool

        """

        self.poolmanager = urllib3.poolmanager.PoolManager(
            num_pools=connections,
            maxsize=maxsize,
            block=block,
            ssl_context=self.ssl_context,
        )


def get_legacy_session():
    """
    Função responsável por abrir a sessão com o certificado
    ssl criando uma autenticação.
    """

    session = requests.session()
    session.mount("https://", CustomHttpAdapter())
    return session


class Regioes(object):

    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo das regiões.
    """

    def __init__(self):
        """
        Faz o request e trás o conteúdo em json.
        """

        url = "https://servicodados.ibge.gov.br/api/v1/localidades/regioes"
        request = get_legacy_session().get(url, headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        return self.json_ibge

    def __repr__(self):
        return repr(str(self.json()))

    def __len__(self):
        return len(self.json_ibge)

    def count(self):
        return len(self.json_ibge)

    def getId(self):
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getSigla(self):
        return [self.json_ibge[i]["sigla"] for i in range(self.count())]

    def getNome(self):
        return [self.json_ibge[i]["nome"] for i in range(self.count())]


class Estados(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo dos Estados.
    """

    def __init__(self, json_ibge=None):
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
        # request = requests.get(url, headers=headers)
        request = get_legacy_session().get(url, headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        return self.json_ibge

    def __repr__(self):
        return repr(str(self.json()))

    def count(self):
        return len(self.json_ibge)

    def getId(self):
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getSigla(self):
        return [self.json_ibge[i]["sigla"] for i in range(self.count())]

    def getNome(self):
        return [self.json_ibge[i]["nome"] for i in range(self.count())]


class Municipios(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo dos Municípios.
    """

    def __init__(self):
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
        # request = requests.get(url, headers=headers)
        request = get_legacy_session().get(url, headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        return self.json_ibge

    def __repr__(self):
        return repr(str(self.json()))

    def count(self):
        return len(self.json_ibge)

    def getId(self):
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getNome(self):
        return [self.json_ibge[i]["nome"] for i in range(self.count())]

    def getDescricaoUF(self):
        return [self.json_ibge[i]["microrregiao"]["mesorregiao"]["UF"]["nome"] for i in range(self.count())]

    def getSiglaUF(self):
        return [self.json_ibge[i]["microrregiao"]["mesorregiao"]["UF"]["sigla"] for i in range(self.count())]

    def getDados(self):
        dados = []
        for i in range(self.count()):
            data = dict()
            data["ibge"] = self.json_ibge[i]["id"]
            data["nome"] = self.json_ibge[i]["nome"]
            data["uf"] = self.json_ibge[i]["microrregiao"]["mesorregiao"]["UF"]["sigla"]
            dados.append(data)
        return dados


class Municipio(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo do Município específico.
    """

    def __init__(self, codigo_ibge=None, json_ibge=None):
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{}"
        # request = requests.get(url.format(codigo_ibge), headers=headers)
        request = get_legacy_session().get(url.format(codigo_ibge), headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        return self.json_ibge

    def __repr__(self):
        return repr(str(self.json()))

    def count(self):
        return int(len(self.json_ibge) / 3)

    def getId(self):
        return self.json_ibge["id"]

    def getNome(self):
        return self.json_ibge["nome"]

    def getDescricaoUF(self):
        return self.json_ibge["microrregiao"]["mesorregiao"]["UF"]["nome"]

    def getUF(self):
        return self.json_ibge["microrregiao"]["mesorregiao"]["UF"]["sigla"]


class MunicipioPorUF(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo do Município por UF.
    """

    def __init__(self, codigo_uf=None):
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/{}/municipios"
        request = get_legacy_session().get(url.format(codigo_uf), headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        return self.json_ibge

    def __repr__(self):
        return repr(str(self.json()))

    def count(self):
        return len(self.json_ibge)

    def getId(self):
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getNome(self):
        return [self.json_ibge[i]["nome"] for i in range(self.count())]
