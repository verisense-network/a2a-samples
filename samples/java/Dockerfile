FROM eclipse-temurin:18-jdk-jammy
WORKDIR /workspace/app
RUN apt-get update

COPY mvnw .
COPY .mvn .mvn
COPY pom.xml .
COPY client client
COPY model model
COPY server server

RUN ./mvnw clean install -DskipTests
EXPOSE 8080 8080

WORKDIR /workspace/app/server

CMD ["../mvnw", "spring-boot:run"]
