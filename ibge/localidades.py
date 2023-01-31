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

        :param Kwargs: Parâmetro extra
        :type: any
        :return: O novo objeto
        :rtype: ssl.SSLContext

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
        :return: O novo objeto
        :rtype: urllib3.poolmanager.PoolManager
        

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
    :return: O novo objeto
    :rtype: requests.sessions.Session

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
        Faz o request e gera o conteúdo em json.

        """

        url = "https://servicodados.ibge.gov.br/api/v1/localidades/regioes"
        request = get_legacy_session().get(url, headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        """
        Retorna o conteúdo.

        :return: String com o conteúdo
        :rtype: str
        """
        return self.json_ibge

    def __repr__(self):
        """
        Representação da api em json.

        :return: String com o conteúdo
        :rtype: str
        """
        return repr(str(self.json()))

    def __len__(self):
        """
        Retorna o tamanho do conteúdo.

        :return: Inteiro reprentando o tamanho do json. 
        :rtype: int
        """
        return len(self.json_ibge)

    def count(self):
        """
        Retorna o tamanho do conteúdo.

        :return: Inteiro reprentando o tamanho do json. 
        :rtype: int
        """
        return len(self.json_ibge)

    def getId(self):
        """
        Retorna os IDs.

        :return: Lista com os ids 
        :rtype: list
        """
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getSigla(self):
        """
        Retorna as siglas.

        :return: Lista com as siglas 
        :rtype: list
        """
        
        return [self.json_ibge[i]["sigla"] for i in range(self.count())]

    def getNome(self):
        """
        Retorna os nomes das regiões.

        :return: Lista com as regiões
        :rtype: list
        """
        return [self.json_ibge[i]["nome"] for i in range(self.count())]


class Estados(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo dos Estados.
    """

    def __init__(self):
        """
        Faz o request e gera o conteúdo em json.

        """
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados"
        request = get_legacy_session().get(url, headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        """
        Retorna o conteúdo.

        :return: String com o conteúdo
        :rtype: str
        """
        return self.json_ibge

    def __repr__(self):
        """
        Representação da api em json.

        :return: String com o conteúdo
        :rtype: str
        """
        return repr(str(self.json()))

    def count(self):
        """
        Retorna o tamanho do conteúdo.

        :return: Inteiro reprentando o tamanho do json. 
        :rtype: int
        """
        return len(self.json_ibge)

    def getId(self):
        """
        Retorna os IDs.

        :return: Lista com os ids 
        :rtype: list
        """
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getSigla(self):
        """
        Retorna as siglas.

        :return: Lista com as siglas 
        :rtype: list
        """
        return [self.json_ibge[i]["sigla"] for i in range(self.count())]

    def getNome(self):
        """
        Retorna os nomes das regiões.

        :return: Lista com as regiões
        :rtype: list
        """
        return [self.json_ibge[i]["nome"] for i in range(self.count())]


class Municipios(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo dos Municípios.
    """

    def __init__(self):
        """
        Faz o request gera o conteúdo em json.

        """
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios"
        # request = requests.get(url, headers=headers)
        request = get_legacy_session().get(url, headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        """
        Retorna o conteúdo.

        :return: String com o conteúdo
        :rtype: str
        """
        return self.json_ibge

    def __repr__(self):
        """
        Representação da api em json.

        :return: String com o conteúdo
        :rtype: str
        """
        return repr(str(self.json()))

    def count(self):
        """
        Retorna o tamanho do conteúdo.

        :return: Inteiro reprentando o tamanho do json. 
        :rtype: int
        """
        return len(self.json_ibge)

    def getId(self):
        """
        Retorna os IDs.

        :return: Lista com os ids 
        :rtype: list
        """
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getNome(self):
        """
        Retorna os nomes dos municípios.

        :return: Lista com as regiões
        :rtype: list
        """
        return [self.json_ibge[i]["nome"] for i in range(self.count())]

    def getDescricaoUF(self):
        """
        Retorna os nomes dos Estados referente aos municípios.

        :return: Lista dos Estados
        :rtype: list
        """
        return [self.json_ibge[i]["microrregiao"]["mesorregiao"]["UF"]["nome"] for i in range(self.count())]

    def getSiglaUF(self):
        """
        Retorna as siglas dos Estados referente aos municípios.

        :return: Lista das siglas dos Estados
        :rtype: list
        """
        return [self.json_ibge[i]["microrregiao"]["mesorregiao"]["UF"]["sigla"] for i in range(self.count())]

    def getDados(self):
        """
        Retorna os dados dos municípios.

        :return: Lista dos dados dos municípios.
        :rtype: list
        """
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

    def __init__(self, codigo_ibge:str=None):
        """
        Faz o request e gera conteúdo em json.
        :param codigo_ibge: Número do código do município.
        :type: str
        """
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/municipios/{}"
        # request = requests.get(url.format(codigo_ibge), headers=headers)
        request = get_legacy_session().get(url.format(codigo_ibge), headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        """
        Retorna o conteúdo.

        :return: Um elemento do dicionário referente ao ID setado.
        :rtype: dict
        """
        return self.json_ibge

    def __repr__(self):
        """
        Representação da api em json.

        :return: String com o conteúdo
        :rtype: str
        """
        return repr(str(self.json()))

    def count(self):
        """
        Retorna o tamanho do conteúdo.

        :return: Inteiro reprentando o tamanho do json. 
        :rtype: int
        """
        return int(len(self.json_ibge) / 3)

    def getId(self):
        """
        Retorna os IDs.

        :return: Lista com o id 
        :rtype: int
        """
        return self.json_ibge["id"]

    def getNome(self):
        """
        Retorna os nomes do município desejado.

        :return: Nome do município
        :rtype: str
        """
        return self.json_ibge["nome"]

    def getDescricaoUF(self):
        """
        Retorna o nome do Estado desejado.

        :return: Nome do Estado
        :rtype: str
        """
        return self.json_ibge["microrregiao"]["mesorregiao"]["UF"]["nome"]

    def getUF(self):
        return self.json_ibge["microrregiao"]["mesorregiao"]["UF"]["sigla"]


class MunicipioPorUF(object):
    """
    Classe que conecta a API IBGE acessando o arquivo
    json trazendo o conteúdo do Município por UF.
    """

    def __init__(self, codigo_uf=None):
        """
        Faz o request e gera o conteúdo em json.
        :param codigo_ibge: Número do código do município.
        :type: str
        """
        url = "https://servicodados.ibge.gov.br/api/v1/localidades/estados/{}/municipios"
        request = get_legacy_session().get(url.format(codigo_uf), headers=headers)
        self.json_ibge = json.loads(request.content.decode("utf-8"))

    def json(self):
        """
        Retorna o conteúdo.

        :return: Lista com o conteúdo
        :rtype: list
        """
        return self.json_ibge

    def __repr__(self):
        """
        Representação da api em json.

        :return: String com o conteúdo
        :rtype: str
        """
        return repr(str(self.json()))

    def count(self):
        """
        Retorna o tamanho do conteúdo.

        :return: Inteiro reprentando o tamanho do json. 
        :rtype: int
        """
        return len(self.json_ibge)

    def getId(self):
        """
        Retorna os IDs.

        :return: Lista com o id 
        :rtype: list
        """
        return [self.json_ibge[i]["id"] for i in range(self.count())]

    def getNome(self):
        """
        Retorna os municípios pelo IDs das UFs.

        :return: Lista com os municípios por UF
        :rtype: list
        """
        return [self.json_ibge[i]["nome"] for i in range(self.count())]
