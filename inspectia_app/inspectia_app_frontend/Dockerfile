# syntax=docker/dockerfile:1
FROM ubuntu:22.04

# Variables d'environnement
ENV DEBIAN_FRONTEND=noninteractive
ENV FLUTTER_VERSION=3.24.0
ENV FLUTTER_HOME=/opt/flutter
ENV PATH=$PATH:$FLUTTER_HOME/bin:$FLUTTER_HOME/bin/cache/dart-sdk/bin

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    curl \
    git \
    unzip \
    xz-utils \
    zip \
    libglu1-mesa \
    lib32stdc++6 \
    lib32z1 \
    libc6-i386 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Installer Flutter
RUN git clone https://github.com/flutter/flutter.git -b stable $FLUTTER_HOME && \
    flutter config --no-analytics && \
    flutter doctor

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers de configuration Flutter
COPY pubspec.yaml pubspec.lock ./

# Installer les dépendances Flutter
RUN flutter pub get

# Copier le code source
COPY . .

# Construire l'application Flutter pour le web
RUN flutter build web --release

# Utiliser nginx pour servir l'application web
FROM nginx:alpine

# Copier les fichiers construits
COPY --from=0 /app/build/web /usr/share/nginx/html

# Copier la configuration nginx personnalisée
COPY nginx.conf /etc/nginx/conf.d/default.conf

# Exposer le port 80
EXPOSE 80

# Commande de démarrage
CMD ["nginx", "-g", "daemon off;"]
