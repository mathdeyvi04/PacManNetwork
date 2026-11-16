# Pac-Man Neural Network 🧠👻

Uma implementação completa do clássico Pac-Man em Rust, onde um agente controla o personagem através de uma Rede Neural (MLP) treinada por um Algoritmo Genético.

O projeto conta com simulação determinística, treinamento paralelo em CPU (Rayon), visualização em tempo real das ativações da rede neural e gravação assíncrona das partidas em vídeo.

## 🚀 Funcionalidades

- **Simulação Completa:** Física do Pac-Man, comportamento clássico dos 4 fantasmas com alternância entre modos Scatter e Chase.
- **Algoritmo Genético:** Evolução com elitismo, crossover uniforme, mutação adaptativa e imigração de redes.
- **Treinamento Paralelo:** Avaliação da população de 100 agentes em múltiplas threads usando `rayon`.
- **Gravação de Vídeo:** Gravação automática da última partida finalizada via `ffmpeg`, utilizando uma thread dedicada (não bloqueia a renderização do jogo).
- **Logs e Visualização:** Gráficos da rede neural em tempo real e logs de evolução (fitness, score, estagnação) salvos em arquivo.

## 🛠️ Requisitos

1. **Rust & Cargo** (edição 2021+).
2. **FFmpeg** instalado e disponível no PATH do sistema (necessário para a gravação de vídeos).
   - *Windows:* `winget install ffmpeg`
   - *Linux:* `sudo apt install ffmpeg`
   - *macOS:* `brew install ffmpeg`

## 🎮 Como Usar

O projeto possui dois binários principais: `train` (treinamento) e `PacManNetwork` (jogar/testar).

### 1. Treinamento (Algoritmo Genético)
Inicia o treinamento da população de 100 agentes por 9999 gerações.
```bash
cargo run --release --bin train
```
*Opções de treinamento:*
- `--seed <u64>`: Define uma seed fixa para reprodutibilidade.
- `--map <caminho.bin>`: Carrega um mapa específico (se não existir, gera e salva um novo).
- `--single` ou `-s`: Força a execução em single-thread (útil para debug).

### 2. Jogar / Testar a Rede Neural
Carrega uma rede neural treinada e a coloca para jogar no mapa especificado.
```bash
cargo run --release --bin pacman -- nn best_gen_XXX.bin map.bin
```
*Opções do modo NN:*
- `nn`: Modo rede neural (se não passar caminhos, usa pesos e mapas aleatórios).
- `human` ou `play`: Joga com o teclado (Setas ou WASD).
- Variável de ambiente `PACMAN_NO_RECORD=1`: Desliga a gravação de vídeo ao rodar a rede.

## 🧠 Arquitetura da Rede Neural

A inteligência do Pac-Man é guiada por um Perceptron de Múltiplas Camadas (MLP - Multilayer Perceptron) fixo.

### Topologia
A rede possui a arquitetura `4 → 16 → 4`:
- **Camada de Entrada:** 4 neurônios (sensores).
- **Camada Oculta:** 16 neurônios com função de ativação **ReLU**.
- **Camada de Saída:** 4 neurônios (lineares), onde o maior valor (argmax) define a ação.

### Sensores (Entradas)
A cada frame, 4 *raycasts* (raios de visão) são disparados a partir do tile atual do Pac-Man nas direções `[Cima, Baixo, Esquerda, Direita]`. O sensor retorna o **primeiro evento significativo** encontrado, seguindo uma ordem de prioridade:
1. `-1.0` **(Fantasma):** Prioridade absoluta (sobrevivência). O raio detecta o fantasma mesmo que haja pastilhas no caminho.
2. `-0.5` **(Parede):** Bloqueia o raio de visão.
3. `0.0` **(Vazio/Nada):** O raio chegou à borda do mapa sem encontrar nada.
4. `+0.5` **(Túnel):** Indica uma passagem segura/teletransporte nas bordas.
5. `+1.0` **(Pastilha):** Recompensa. O primeiro pellet visto antes de bater em uma parede.

### Ações (Saídas)
O neurônio com maior ativação dita a direção do próximo movimento:
- `0`: Cima
- `1`: Baixo
- `2`: Esquerda
- `3`: Direita

**Proteção Anti-Jitter:** Para evitar que a rede fique alternando entre direções opostas repetidamente, há um *cooldown* de 8 frames que ignora comandos de inversão brusca de rota.

### Algoritmo Genético (Como ela aprende)
A rede não usa *backpropagation* (gradiente descendente). Ela evolui via Algoritmo Genético:
- **Fitness:** Avaliada pela média de 5 partidas com sementes fixas. Leva em conta a pontuação (pellet), porcentagem de conclusão do mapa, penalidade por passos dados e bônus/penalidades de vitória/morte.
- **Crossover Uniforme:** Cada peso (gene) do filho tem 50% de chance de vir do pai A ou do pai B.
- **Mutação Adaptativa:** Se o treinamento estagnar por 10 gerações, a taxa e a força da mutação aumentam temporariamente para forçar a rede a sair de ótimos locais.
- **Elitismo:** Os 4 melhores agentes de cada geração passam intactos para a próxima.

---

## 🔬 Área para Futuras Implementações de Redes
*(Espaço reservado para documentar testes com novas arquiteturas)*

- [ ] **[Nome da Rede 1] - (Ex: Rede Recorrente/LSTM):** *Descrever como a memória do frame anterior influenciou a tomada de decisão.*
- [ ] **[Nome da Rede 2] - (Ex: CNN com Visão Computacional):** *Descrever os testes usando uma matriz do grid inteiro (one-hot encoding) como entrada, em vez de raycasts.*
- [ ] **[Nome da Rede 3] - (Ex: DQN - Deep Q-Learning):** *Descrever a transição de AG para Reinforcement Learning (RL) e a implementação de uma Replay Memory.*
- [ ] **[Nome da Rede 4] - (Ex: NEAT - NeuroEvolution of Augmenting Topologies):** *Descrever a mudança de topologia fixa para uma topologia evolutiva.*

---

## 📤 Saídas Geradas

- `train_historic.txt`: Histórico completo do último treinamento (gerações, fitness, tempo).
- `map.bin`: O mapa atual (gerado ou carregado).
- `best_gen_XXX.bin`: A melhor rede neural salva ao fim do treinamento.
- `recordings/latest_match.mp4`: O vídeo da última partida completa jogada pela rede neural no modo `nn`.
```