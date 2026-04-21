FROM postgres:18

# Install build tools and PostgreSQL server dev headers
RUN apt-get update && apt-get install -y \
    libnlopt-dev \
    postgresql-server-dev-18 \
    make \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy your extension source code
COPY pg_forecast /usr/src/pg_forecast

# Build and install the extension
WORKDIR /usr/src/pg_forecast/src/pg_forecast
RUN make && make install && make clean

# Copy init script to automatically create extension
COPY init-extension.sh /docker-entrypoint-initdb.d/init-extension.sh

# Ensure script is executable
RUN chmod +x /docker-entrypoint-initdb.d/init-extension.sh