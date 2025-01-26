#include <iostream>
#include <chrono>
#include <vector>
#include <x86intrin.h>
#include <iomanip> // Para std::fixed y std::setprecision
#include <fstream>
#include <cstdlib> // Para system()

int main()
{
    /**
     * Data-Oriented Programming vs Object-Oriented Programming
     * @see Mike Acton CppCon 2014
     */
    const int num_entities = 1000000;

    /// Clase OOP con mal orden de atributos, causando padding
    /// El padding será usado para alinear correctamente los atributos en memoria
    /// El orden de los atributos es aleatorio y no toma en cuenta el tamaño de cada tipo de dato.
    /// Esto causa que el compilador agregue padding (bytes de relleno) para alinear correctamente
    /// los atributos en la memoria. Como resultado, se desperdicia memoria.
    std::chrono::duration<double> elapsedOOPBad;
    class Entity_OOP_Bad {
    public:
        struct atributes {
            double dx, dz;       // 8 bytes cada uno (16 bytes en total)
            float x, y;          // 4 bytes cada uno (8 bytes)
            uint16_t something;  // 2 bytes
            double dy;           // 8 bytes
            uint16_t something1; // 2 bytes
            uint16_t something2; // 2 bytes
            int score;           // 4 bytes
            int score1;          // 4 bytes
            int score2;          // 4 bytes
            char id;             // 1 byte
            float z;             // 4 bytes
            bool active;         // 1 byte
                                 // _______
                                 // 56 bytes total, alignment 8 bytes
        };

        atributes mAtributes;

        void modifyParams(){
            this->mAtributes.x = this->mAtributes.y = this->mAtributes.z = 0.0f;
            this->mAtributes.dx = this->mAtributes.dy = this->mAtributes.dz = 0.1;
            this->mAtributes.active = true;
            this->mAtributes.id = 'A';
            this->mAtributes.score = 100;
            this->mAtributes.score1 = 100;
            this->mAtributes.score2 = 100;
            this->mAtributes.something *= 2;
            this->mAtributes.something1 *= 2;
            this->mAtributes.something2 *= 2;
        }
    };
    {
        std::vector<Entity_OOP_Bad> entities(num_entities);
        auto start = std::chrono::high_resolution_clock::now();
        unsigned long long start_cycles = __rdtsc();
        for (auto& entity : entities) entity.modifyParams();
        unsigned long long end_cycles = __rdtsc();
        elapsedOOPBad = std::chrono::high_resolution_clock::now() - start;
        std::cout << "OOP (Bad Order) CPU cycles: " << (end_cycles - start_cycles) << "\n";
        std::cout << "OOP (Bad Order) Execution time: " << elapsedOOPBad.count() << " seconds\n";
    }

    /// Clase OOP con buen orden de atributos, minimizando el padding
    /// Aquí se reduce el padding al agrupar los tipos similares
    /// Aquí los atributos se reorganizan de mayor a menor tamaño (primero double, luego float, después int, char, y finalmente bool).
    /// Esto minimiza la cantidad de padding necesario, haciendo que la estructura sea más compacta en memoria.
    /// A un nivel más técnico, al realizar operaciones en los atributos, el código máquina hará búsquedas de registros a partir
    /// del rax (rax+4, rax+20...) con menos desplazamientos, y por tanto más eficientemente, si los atributos están bien ordenados
    std::chrono::duration<double> elapsedOOPDOP;
    class Entity_OOP_Good {
    public:
        struct atributes {
            double dx, dy, dz;   // 8 bytes cada uno (24 bytes en total)
            float x, y, z;       // 4 bytes cada uno (12 bytes)
            int score;           // 4 bytes
            int score1;          // 4 bytes
            int score2;          // 4 bytes
            uint16_t something;  // 2 bytes
            uint16_t something1; // 2 bytes
            uint16_t something2; // 2 bytes
            char id;             // 1 byte
            bool active;         // 1 byte
                                 // _______
                                 // 56 bytes total, alignment 8 bytes
        };

        atributes mAtributes;

        void modifyParams(){
            this->mAtributes.x = this->mAtributes.y = this->mAtributes.z = 0.0f;
            this->mAtributes.dx = this->mAtributes.dy = this->mAtributes.dz = 0.1;
            this->mAtributes.active = true;
            this->mAtributes.id = 'A';
            this->mAtributes.score = 100;
            this->mAtributes.score1 = 100;
            this->mAtributes.score2 = 100;
            this->mAtributes.something *= 2;
            this->mAtributes.something1 *= 2;
            this->mAtributes.something2 *= 2;
        }
    };
    {
        std::vector<Entity_OOP_Good> entities(num_entities);
        auto start = std::chrono::high_resolution_clock::now();
        unsigned long long start_cycles = __rdtsc();
        for (auto& entity : entities) entity.modifyParams();
        unsigned long long end_cycles = __rdtsc();
        elapsedOOPDOP = std::chrono::high_resolution_clock::now() - start;
        std::cout << "OOP (Good Order by DOP) CPU cycles: " << (end_cycles - start_cycles) << "\n";
        std::cout << "OOP (Good Order by DOP) Execution time: " << elapsedOOPDOP.count() << " seconds\n";
    }

    /// Con el comando $ lscpu se puede ver la información de mi CPU, para ver el tamaño en bytes que la CPU consulta en cada ciclo, para
    /// saber cómo maximizar la eficiencia de mi estructura para que no haya huecos innecesarios y hacer las operaciones en el menos número
    /// de ciclos (tamaños cachés L1 y L2, tamaño del bus de datos de 64 bits...)
    ///
    std::chrono::duration<double> elapsedOOPDOP_GoodWithFooPadding;
    class Entity_OOP_GoodWithFooPadding {
    public:
        struct atributes {
            double dx, dy, dz;   // 8 bytes cada uno (24 bytes en total)
            float x, y, z;       // 4 bytes cada uno (12 bytes)
            int score;           // 4 bytes
            int score1;          // 4 bytes
            int score2;          // 4 bytes
            uint16_t something;  // 2 bytes
            uint16_t something1; // 2 bytes
            uint16_t something2; // 2 bytes
            char id;             // 1 byte
            bool active;         // 1 byte
            char padding[8];     // 8 bytes de padding para completar el bloque de 64 bytes
                                 // _______
                                 // 64 bytes total, alignment 8 bytes
        };

        atributes mAtributes;

        void modifyParams() {
            this->mAtributes.x = this->mAtributes.y = this->mAtributes.z = 0.0f;
            this->mAtributes.dx = this->mAtributes.dy = this->mAtributes.dz = 0.1;
            this->mAtributes.active = true;
            this->mAtributes.id = 'A';
            this->mAtributes.score = 100;
            this->mAtributes.score1 = 100;
            this->mAtributes.score2 = 100;
            this->mAtributes.something *= 2;
            this->mAtributes.something1 *= 2;
            this->mAtributes.something2 *= 2;
        }
    };
    {
        std::vector<Entity_OOP_GoodWithFooPadding> entities(num_entities);
        auto start = std::chrono::high_resolution_clock::now();
        unsigned long long start_cycles = __rdtsc();
        for (auto& entity : entities) entity.modifyParams();
        unsigned long long end_cycles = __rdtsc();
        elapsedOOPDOP_GoodWithFooPadding = std::chrono::high_resolution_clock::now() - start;
        std::cout << "OOP (Good Order by DOP and Foo Padding) CPU cycles: " << (end_cycles - start_cycles) << "\n";
        std::cout << "OOP (Good Order by DOP and Foo Padding) Execution time: " << elapsedOOPDOP_GoodWithFooPadding.count() << " seconds\n";
    }

    std::cout << "With DOP, the processing is " << (elapsedOOPBad.count() - elapsedOOPDOP.count()) * 1e3 << " ms faster\n";
    std::cout << "With DOP and Foo Padding, the processing is " << (elapsedOOPBad.count() - elapsedOOPDOP_GoodWithFooPadding.count()) * 1e3 << " ms faster\n";

    /// Uso de RDTSC para contar ciclos de CPU
    {
        //            const int num_entities = 1000;
        //            std::vector<float> data(num_entities, 1.0f);
        //            unsigned long long start_cycles = __rdtsc();
        //            for (int i = 0; i < num_entities; ++i) {
        //                data[i] *= 1.1f;
        //            }
        //            unsigned long long end_cycles = __rdtsc();
        //            std::cout << "CPU cycles: " << (end_cycles - start_cycles) << "\n";
    }

    /// Las CPU modernas acceden a la memoria en bloques (típicamente de 8 bytes o más). Si los datos están correctamente alineados
    /// en la memoria, el acceso es más rápido porque puede cargar y almacenar los datos en un solo ciclo de memoria
    /// Si los datos no están alineados correctamente, la CPU puede tener que realizar más accesos a memoria, lo que introduce
    /// penalizaciones de rendimiento debido a la necesidad de corregir la alineación en tiempo de ejecución.

    {
        const int num_trials = 1000; // Número de veces que se va a ejecutar todo esto para ver si no es casual la rapidez
        std::vector<int> rankingOOPBadTime, rankingOOPDOPTime, rankingOOPDOP_GoodWithFooPaddingTime;
        std::vector<int> rankingOOPBadCycles, rankingOOPDOPCycles, rankingOOPDOP_GoodWithFooPaddingCycles;
        double avgOOPBadTime = 0.0, avgOOPDOPTime = 0.0, avgOOPDOP_GoodWithFooPaddingTime = 0.0;
        unsigned long long avgOOPBadCycles = 0, avgOOPDOPCycles = 0, avgOOPDOP_GoodWithFooPaddingCycles = 0;

        std::ofstream outputFile("../../benchmark_results.txt");
        if (!outputFile.is_open()) {
            std::cerr << "Error al abrir benchmark_results.\n";
            return 1;
        }

        std::ofstream averagesOutputFile("../../averages_results.txt");
        if (!outputFile.is_open()) {
            std::cerr << "Error al abrir averages_results.\n";
            return 1;
        }

        std::chrono::duration<double> duration;
        for (int i = 0; i < num_trials; ++i)
        {
            std::vector<Entity_OOP_Bad> entities(num_entities);
            auto start = std::chrono::high_resolution_clock::now();
            unsigned long long start_cycles = __rdtsc();
            for (auto& entity : entities) entity.modifyParams();
            unsigned long long end_cycles = __rdtsc();
            duration = std::chrono::high_resolution_clock::now() - start;
            double timeOOPBad = duration.count();
            unsigned long long cyclesOOPBad = end_cycles - start_cycles;

            std::vector<Entity_OOP_Good> entities2(num_entities);
            start = std::chrono::high_resolution_clock::now();
            start_cycles = __rdtsc();
            for (auto& entity : entities2) entity.modifyParams();
            end_cycles = __rdtsc();
            duration = std::chrono::high_resolution_clock::now() - start;
            double timeOOPDOP = duration.count();
            unsigned long long cyclesOOPDOP = end_cycles - start_cycles;

            std::vector<Entity_OOP_GoodWithFooPadding> entities3(num_entities);
            start = std::chrono::high_resolution_clock::now();
            start_cycles = __rdtsc();
            for (auto& entity : entities3) entity.modifyParams();
            end_cycles = __rdtsc();
            duration = std::chrono::high_resolution_clock::now() - start;
            double timeOOPDOP_GoodWithFooPadding = duration.count();
            unsigned long long cyclesOOPDOP_GoodWithFooPadding = end_cycles - start_cycles;

            avgOOPBadTime += timeOOPBad;
            avgOOPDOPTime += timeOOPDOP;
            avgOOPDOP_GoodWithFooPaddingTime += timeOOPDOP_GoodWithFooPadding;
            avgOOPBadCycles += cyclesOOPBad;
            avgOOPDOPCycles += cyclesOOPDOP;
            avgOOPDOP_GoodWithFooPaddingCycles += cyclesOOPDOP_GoodWithFooPadding;

            // Ranking por tiempo (menor es más rápido)
            std::vector<std::pair<std::string, double>> timeResults = {
                {"OOP (Bad Order)", timeOOPBad},
                {"OOP (Good Order by DOP)", timeOOPDOP},
                {"OOP (Good Order by DOP and Foo Padding)", timeOOPDOP_GoodWithFooPadding}
            };
            std::sort(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& a, const std::pair<std::string, double>& b) {
                return a.second < b.second;
            });

            // Asignar el ranking por tiempo
            rankingOOPBadTime.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "OOP (Bad Order)"; })) + 1);
            rankingOOPDOPTime.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "OOP (Good Order by DOP)"; })) + 1);
            rankingOOPDOP_GoodWithFooPaddingTime.push_back(std::distance(timeResults.begin(), std::find_if(timeResults.begin(), timeResults.end(), [](const std::pair<std::string, double>& p) { return p.first == "OOP (Good Order by DOP and Foo Padding)"; })) + 1);

            // Ranking por ciclos (menor es más rápido)
            std::vector<std::pair<std::string, unsigned long long>> cycleResults = {
                {"OOP (Bad Order)", cyclesOOPBad},
                {"OOP (Good Order by DOP)", cyclesOOPDOP},
                {"OOP (Good Order by DOP and Foo Padding)", cyclesOOPDOP_GoodWithFooPadding}
            };
            std::sort(cycleResults.begin(), cycleResults.end(), [](const std::pair<std::string, unsigned long long>& a, const std::pair<std::string, unsigned long long>& b) {
                return a.second < b.second;
            });

            // Asignar el ranking por ciclos
            rankingOOPBadCycles.push_back(std::distance(cycleResults.begin(), std::find_if(cycleResults.begin(), cycleResults.end(), [](const std::pair<std::string, unsigned long long>& p) { return p.first == "OOP (Bad Order)"; })) + 1);
            rankingOOPDOPCycles.push_back(std::distance(cycleResults.begin(), std::find_if(cycleResults.begin(), cycleResults.end(), [](const std::pair<std::string, unsigned long long>& p) { return p.first == "OOP (Good Order by DOP)"; })) + 1);
            rankingOOPDOP_GoodWithFooPaddingCycles.push_back(std::distance(cycleResults.begin(), std::find_if(cycleResults.begin(), cycleResults.end(), [](const std::pair<std::string, unsigned long long>& p) { return p.first == "OOP (Good Order by DOP and Foo Padding)"; })) + 1);
        }

        avgOOPBadTime /= num_trials;
        avgOOPDOPTime /= num_trials;
        avgOOPDOP_GoodWithFooPaddingTime /= num_trials;
        avgOOPBadCycles /= num_trials;
        avgOOPDOPCycles /= num_trials;
        avgOOPDOP_GoodWithFooPaddingCycles /= num_trials;

        // Guardar los promedios en un archivo
        averagesOutputFile << "Promedio_Tiempo_OOP_Bad, Promedio_Ciclos_OOP_Bad, "
            << "Promedio_Tiempo_OOP_Good_DOP, Promedio_Ciclos_OOP_Good_DOP, "
            << "Promedio_Tiempo_OOP_Good_FooPadding, Promedio_Ciclos_OOP_Good_FooPadding\n";
        averagesOutputFile << std::fixed << std::setprecision(6);
        averagesOutputFile << avgOOPBadTime << ", "
                   << avgOOPBadCycles << ", "
                   << avgOOPDOPTime << ", "
                   << avgOOPDOPCycles << ", "
                   << avgOOPDOP_GoodWithFooPaddingTime << ", "
                   << avgOOPDOP_GoodWithFooPaddingCycles << "\n";
        averagesOutputFile.close();

        // Guardar los rankings en un archivo
        for (int i = 0; i < num_trials; ++i) {
            outputFile << i + 1 << ", "
                       << rankingOOPBadTime[i] << ", "
                       << rankingOOPDOPTime[i] << ", "
                       << rankingOOPDOP_GoodWithFooPaddingTime[i] << ", "
                       << rankingOOPBadCycles[i] << ", "
                       << rankingOOPDOPCycles[i] << ", "
                       << rankingOOPDOP_GoodWithFooPaddingCycles[i] << "\n";
        }
        outputFile.close();
    }

    return 0;
}
