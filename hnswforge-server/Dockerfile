FROM eclipse-temurin:25-jdk
WORKDIR /app
COPY target/hnswforge-server-0.1.0-SNAPSHOT.jar /app/hnswforge-server.jar
EXPOSE 8080
ENTRYPOINT ["java","--add-modules","jdk.incubator.vector","-jar","/app/hnswforge-server.jar"]
