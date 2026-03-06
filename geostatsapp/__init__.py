# Importa os módulos para que fiquem acessíveis se o usuário quiser usar a matemática pura
from . import gslib
from . import utils
from . import plots
from . import variografia

# Importa o App principal do dashboard para facilitar a chamada no Colab
from .dashboard import App

__version__ = '0.1.0'