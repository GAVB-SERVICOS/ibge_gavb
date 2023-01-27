from ibge.localidades import CustomHttpAdapter
from ibge.localidades import get_legacy_session
from unittest.mock import Mock, patch
from ibge.localidades import Regioes, Estados, Municipio, Municipios, MunicipioPorUF


def test_CustomHttpAdapter():
    adapter = CustomHttpAdapter()
    assert adapter.ssl_context.options == 2186412244


def test_get_legacy_session():
    session = get_legacy_session()
    assert isinstance(session.adapters['https://'], CustomHttpAdapter)


@patch('ibge.localidades.get_legacy_session')
def test_Regioes(get_mock):

    get_mock.return_value.get.return_value = Mock()
    get_mock.return_value.get.return_value.content = b'[{"id":1,"sigla":"N","nome":"Norte"},{"id":2,"sigla":"NE","nome":"Nordeste"},{"id":3,"sigla":"SE","nome":"Sudeste"},{"id":4,"sigla":"S","nome":"Sul"},{"id":5,"sigla":"CO","nome":"Centro-Oeste"}]'

    regioes = Regioes()
    json = regioes.json()
    id = regioes.getId()
    sigla = regioes.getSigla()
    count = regioes.count()
    nome = regioes.getNome()

    assert json == [
        {'id': 1, 'sigla': 'N', 'nome': 'Norte'},
        {'id': 2, 'sigla': 'NE', 'nome': 'Nordeste'},
        {'id': 3, 'sigla': 'SE', 'nome': 'Sudeste'},
        {'id': 4, 'sigla': 'S', 'nome': 'Sul'},
        {'id': 5, 'sigla': 'CO', 'nome': 'Centro-Oeste'},
    ]
    assert count == 5
    assert id == [1, 2, 3, 4, 5]
    assert sigla == ['N', 'NE', 'SE', 'S', 'CO']
    assert nome == ['Norte', 'Nordeste', 'Sudeste', 'Sul', 'Centro-Oeste']


@patch('ibge.localidades.get_legacy_session')
def test_Estados(get_mock):

    get_mock.return_value.get.return_value = Mock()
    get_mock.return_value.get.return_value.content = b'[{"id":11,"sigla":"RO","nome":"Rond\xc3\xb4nia","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":12,"sigla":"AC","nome":"Acre","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":13,"sigla":"AM","nome":"Amazonas","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":14,"sigla":"RR","nome":"Roraima","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":15,"sigla":"PA","nome":"Par\xc3\xa1","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":16,"sigla":"AP","nome":"Amap\xc3\xa1","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":17,"sigla":"TO","nome":"Tocantins","regiao":{"id":1,"sigla":"N","nome":"Norte"}},{"id":21,"sigla":"MA","nome":"Maranh\xc3\xa3o","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":22,"sigla":"PI","nome":"Piau\xc3\xad","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":23,"sigla":"CE","nome":"Cear\xc3\xa1","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":24,"sigla":"RN","nome":"Rio Grande do Norte","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":25,"sigla":"PB","nome":"Para\xc3\xadba","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":26,"sigla":"PE","nome":"Pernambuco","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":27,"sigla":"AL","nome":"Alagoas","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":28,"sigla":"SE","nome":"Sergipe","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":29,"sigla":"BA","nome":"Bahia","regiao":{"id":2,"sigla":"NE","nome":"Nordeste"}},{"id":31,"sigla":"MG","nome":"Minas Gerais","regiao":{"id":3,"sigla":"SE","nome":"Sudeste"}},{"id":32,"sigla":"ES","nome":"Esp\xc3\xadrito Santo","regiao":{"id":3,"sigla":"SE","nome":"Sudeste"}},{"id":33,"sigla":"RJ","nome":"Rio de Janeiro","regiao":{"id":3,"sigla":"SE","nome":"Sudeste"}},{"id":35,"sigla":"SP","nome":"S\xc3\xa3o Paulo","regiao":{"id":3,"sigla":"SE","nome":"Sudeste"}},{"id":41,"sigla":"PR","nome":"Paran\xc3\xa1","regiao":{"id":4,"sigla":"S","nome":"Sul"}},{"id":42,"sigla":"SC","nome":"Santa Catarina","regiao":{"id":4,"sigla":"S","nome":"Sul"}},{"id":43,"sigla":"RS","nome":"Rio Grande do Sul","regiao":{"id":4,"sigla":"S","nome":"Sul"}},{"id":50,"sigla":"MS","nome":"Mato Grosso do Sul","regiao":{"id":5,"sigla":"CO","nome":"Centro-Oeste"}},{"id":51,"sigla":"MT","nome":"Mato Grosso","regiao":{"id":5,"sigla":"CO","nome":"Centro-Oeste"}},{"id":52,"sigla":"GO","nome":"Goi\xc3\xa1s","regiao":{"id":5,"sigla":"CO","nome":"Centro-Oeste"}},{"id":53,"sigla":"DF","nome":"Distrito Federal","regiao":{"id":5,"sigla":"CO","nome":"Centro-Oeste"}}]'

    estados = Estados()
    json = estados.json()
    id = estados.getId()
    sigla = estados.getSigla()
    count = estados.count()
    nome = estados.getNome()

    assert json[0] == {"id": 11, "sigla": "RO", "nome": "Rondônia", "regiao": {"id": 1, "sigla": "N", "nome": "Norte"}}
    assert count == 27
    assert id[0] == 11
    assert sigla[0] == 'RO'


@patch('ibge.localidades.get_legacy_session')
def test_Municipios(get_mock):

    get_mock.return_value.get.return_value = Mock()
    get_mock.return_value.get.return_value.content = b'[{"id":1100015,"nome":"Alta Floresta D\'Oeste","microrregiao":{"id":11006,"nome":"Cacoal","mesorregiao":{"id":1102,"nome":"Leste Rondoniense","UF":{"id":11,"sigla":"RO","nome":"Rond\xc3\xb4nia","regiao":{"id":1,"sigla":"N","nome":"Norte"}}}},"regiao-imediata":{"id":110005,"nome":"Cacoal","regiao-intermediaria":{"id":1102,"nome":"Ji-Paran\xc3\xa1","UF":{"id":11,"sigla":"RO","nome":"Rond\xc3\xb4nia","regiao":{"id":1,"sigla":"N","nome":"Norte"}}}}}]'

    municipios = Municipios()
    # assert get_mock.called
    json = municipios.json()
    id = municipios.getId()
    count = municipios.count()
    nome = municipios.getNome()
    descricao = municipios.getDescricaoUF()
    dados = municipios.getDados()

    assert json[0] == {
        'id': 1100015,
        'nome': "Alta Floresta D'Oeste",
        'microrregiao': {
            'id': 11006,
            'nome': 'Cacoal',
            'mesorregiao': {
                'id': 1102,
                'nome': 'Leste Rondoniense',
                'UF': {
                    'id': 11,
                    'sigla': 'RO',
                    'nome': 'Rondônia',
                    'regiao': {'id': 1, 'sigla': 'N', 'nome': 'Norte'},
                },
            },
        },
        'regiao-imediata': {
            'id': 110005,
            'nome': 'Cacoal',
            'regiao-intermediaria': {
                'id': 1102,
                'nome': 'Ji-Paraná',
                'UF': {
                    'id': 11,
                    'sigla': 'RO',
                    'nome': 'Rondônia',
                    'regiao': {'id': 1, 'sigla': 'N', 'nome': 'Norte'},
                },
            },
        },
    }
    assert count == 1  # 5570
    assert id[0] == 1100015
    assert nome[0] == "Alta Floresta D'Oeste"
    assert descricao[0] == 'Rondônia'
    assert dados[0] == {'ibge': 1100015, 'nome': "Alta Floresta D'Oeste", 'uf': 'RO'}


@patch('ibge.localidades.get_legacy_session')
def test_Municipio(get_mock):

    get_mock.return_value.get.return_value = Mock()
    get_mock.return_value.get.return_value.content = b'{"id":5221858,"nome":"Valpara\xc3\xadso de Goi\xc3\xa1s","microrregiao":{"id":52012,"nome":"Entorno de Bras\xc3\xadlia","mesorregiao":{"id":5204,"nome":"Leste Goiano","UF":{"id":52,"sigla":"GO","nome":"Goi\xc3\xa1s","regiao":{"id":5,"sigla":"CO","nome":"Centro-Oeste"}}}},"regiao-imediata":{"id":520019,"nome":"Luzi\xc3\xa2nia","regiao-intermediaria":{"id":5206,"nome":"Luzi\xc3\xa2nia - \xc3\x81guas Lindas de Goi\xc3\xa1s","UF":{"id":52,"sigla":"GO","nome":"Goi\xc3\xa1s","regiao":{"id":5,"sigla":"CO","nome":"Centro-Oeste"}}}}}'

    municipio = Municipio('5221858')

    count = municipio.count()
    json = municipio.json()
    id = municipio.getId()
    nome = municipio.getNome()
    uf = municipio.getUF()
    descricao = municipio.getDescricaoUF()
    assert count == 1
    assert json == {
        'id': 5221858,
        'nome': 'Valparaíso de Goiás',
        'microrregiao': {
            'id': 52012,
            'nome': 'Entorno de Brasília',
            'mesorregiao': {
                'id': 5204,
                'nome': 'Leste Goiano',
                'UF': {
                    'id': 52,
                    'sigla': 'GO',
                    'nome': 'Goiás',
                    'regiao': {'id': 5, 'sigla': 'CO', 'nome': 'Centro-Oeste'},
                },
            },
        },
        'regiao-imediata': {
            'id': 520019,
            'nome': 'Luziânia',
            'regiao-intermediaria': {
                'id': 5206,
                'nome': 'Luziânia - Águas Lindas de Goiás',
                'UF': {
                    'id': 52,
                    'sigla': 'GO',
                    'nome': 'Goiás',
                    'regiao': {'id': 5, 'sigla': 'CO', 'nome': 'Centro-Oeste'},
                },
            },
        },
    }

    assert id == 5221858
    assert nome == 'Valparaíso de Goiás'
    assert uf == 'GO'
    assert descricao == 'Goiás'


@patch('ibge.localidades.get_legacy_session')
def test_MunicipioPorUF(get_mock):

    get_mock.return_value.get.return_value = Mock()
    get_mock.return_value.get.return_value.content = b'[{"id":1300029,"nome":"Alvar\xc3\xa3es","microrregiao":{"id":13005,"nome":"Tef\xc3\xa9","mesorregiao":{"id":1303,"nome":"Centro Amazonense","UF":{"id":13,"sigla":"AM","nome":"Amazonas","regiao":{"id":1,"sigla":"N","nome":"Norte"}}}},"regiao-imediata":{"id":130005,"nome":"Tef\xc3\xa9","regiao-intermediaria":{"id":1302,"nome":"Tef\xc3\xa9","UF":{"id":13,"sigla":"AM","nome":"Amazonas","regiao":{"id":1,"sigla":"N","nome":"Norte"}}}}}]'

    municipiosUF = MunicipioPorUF('13')
    json = municipiosUF.json()
    count = municipiosUF.count()
    id = municipiosUF.getId()
    nome = municipiosUF.getNome()

    assert json[0] == {
        'id': 1300029,
        'nome': 'Alvarães',
        'microrregiao': {
            'id': 13005,
            'nome': 'Tefé',
            'mesorregiao': {
                'id': 1303,
                'nome': 'Centro Amazonense',
                'UF': {
                    'id': 13,
                    'sigla': 'AM',
                    'nome': 'Amazonas',
                    'regiao': {'id': 1, 'sigla': 'N', 'nome': 'Norte'},
                },
            },
        },
        'regiao-imediata': {
            'id': 130005,
            'nome': 'Tefé',
            'regiao-intermediaria': {
                'id': 1302,
                'nome': 'Tefé',
                'UF': {
                    'id': 13,
                    'sigla': 'AM',
                    'nome': 'Amazonas',
                    'regiao': {'id': 1, 'sigla': 'N', 'nome': 'Norte'},
                },
            },
        },
    }
    assert count == 1  # 62
    assert id[0] == 1300029
    assert nome[0] == 'Alvarães'
