#include <iostream>
#include <vector>
#include <array>
#include <cmath>
#include <optional>
#include <tuple>
#include <unordered_set>
#include <map>
#include <Eigen/Dense>

template<size_t d, size_t n>
class Mesh;

template<size_t d, size_t n>
class MeshSimplex;

template<size_t d, size_t n>
class MeshSimplices;

size_t BinomialCoefficient(size_t n, size_t k){
    size_t coefficient = 1;
    for(size_t i = 1; i <= k; i++){
        coefficient *= (n + 1 - i) / i;
    }
    return coefficient;
}

template<typename T>
class Subsets{
    private:
        std::vector<T> set;
        size_t size;
    
    public:
        Subsets(std::vector<T> set, size_t size){
            this->set = set;
            this->size = size;
        }

        struct Iterator {
            const Subsets* parent;
            std::vector<size_t> indices;
            bool is_finished;

            Iterator(const Subsets* parent, std::vector<size_t> indices, bool is_finished){
                this->parent = parent;
                this->indices = indices;
                this->is_finished = is_finished;
            }

            const std::vector<T> operator*() const{
                std::vector<T> subset(this->indices.size());
                for(size_t i = 0; i < indices.size(); i++){
                    subset[i] = this->parent->set[this->indices[i]];
                }
                return subset;
            }

            Iterator& operator++() {
                size_t k = this->parent->size;
                for (int i = k - 1; i >= 0; --i) {
                    if (indices[i] > k - i - 1) {
                        indices[i]--;
                        for (size_t j = i + 1; j < k; ++j) {
                            indices[j] = indices[j - 1] - 1;
                        }
                        return *this;
                    }
                }
                this->is_finished = true;
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                return this->is_finished != other.is_finished;
            }
        };

        Iterator begin() const{
            size_t n = this->set.size();
            size_t k = this->size;
            std::vector<size_t> first_indices(k);
            for (size_t i = 0; i < k; i++) {
                first_indices[i] = n - 1 - i;
            }
            return Iterator(this, first_indices, false);
        }

        Iterator end() const{
            return Iterator(this, std::vector<size_t>(this->size), true);
        }
};

class BooleanMatrixCSR{
    private:
        int number_of_columns;
        std::vector<size_t> offsets;
        std::vector<size_t> indices;

        BooleanMatrixCSR(
            int number_of_columns,
            std::vector<size_t> offsets,
            std::vector<size_t> indices
        ){
            this->number_of_columns = number_of_columns;
            this->offsets = offsets;
            this->indices = indices;
        }

        template<size_t d, size_t n>
        friend class Mesh;

    public:
        size_t NumberOfRows() const{
            int n = this->offsets.size() - 1;
            return n;
        }

        size_t NumberOfColumns() const{
            int m = this->number_of_columns;
            return m;
        }

        std::vector<size_t> operator[](size_t i) const{
            size_t size = this->offsets[i + 1] - this->offsets[i];
            std::vector<size_t> indices(size);
            for(size_t j = 0; j < size; j++){
                indices[j] = this->indices[this->offsets[i] + j];
            }
            return indices;
        }

        BooleanMatrixCSR Transpose(){
            size_t n = this->NumberOfColumns();
            size_t m = this->NumberOfRows();

            std::vector<size_t> B_degrees(n, 0);
            for(size_t j = 0; j < m; j++){
                for(auto i : (*this)[j]){
                    B_degrees[i] += 1;
                }
            }

            std::vector<size_t> B_offsets(n + 1);
            B_offsets[0] = 0;
            for(size_t i = 0; i < n; i++){
                B_offsets[i + 1] = B_offsets[i] + B_degrees[i];
            }

            std::vector<size_t> B_indices(B_offsets.back());
            std::vector<size_t> cursors = B_offsets;
            for(size_t j = 0; j < m; j++){
                for(auto i : (*this)[j]){
                    B_indices[cursors[i]] = j;
                    cursors[i] += 1;
                }
            }

            return BooleanMatrixCSR(m, B_offsets, B_indices);
        }

        friend BooleanMatrixCSR Compose(const BooleanMatrixCSR& A, const BooleanMatrixCSR& B);
        friend BooleanMatrixCSR BooleanIdentity(size_t size);
};

BooleanMatrixCSR BooleanIdentity(size_t size){
    std::vector<size_t> offsets(size + 1);
    std::vector<size_t> indices(size);
    for(size_t i = 0; i < size; i++){
        offsets[i] = i;
        indices[i] = i;
    }
    offsets[size] = size;
    return BooleanMatrixCSR(size, offsets, indices);
}

BooleanMatrixCSR Compose(const BooleanMatrixCSR& A, const BooleanMatrixCSR& B){
    size_t n = A.NumberOfRows();
    size_t m = B.NumberOfColumns();

    std::vector<size_t> C_offsets(n + 1, 0);
    std::vector<std::vector<size_t>> columns(n);
    for(size_t i = 0; i < n; i++){
        std::unordered_set<size_t> temp;
        for(auto j : A[i]){
            for(auto k : B[j]){
                temp.insert(k);
            }
        }

        columns[i].assign(temp.begin(), temp.end());
        std::sort(columns[i].begin(), columns[i].end());
        C_offsets[i+1] = C_offsets[i] + columns[i].size();
    }

    std::vector<size_t> C_indices(C_offsets.back());
    size_t pos = 0;
    for(size_t i = 0; i < n; i++){
        for(auto k : columns[i]){
            C_indices[pos] = k;
            pos++;
        }
    }

    return BooleanMatrixCSR(m, C_offsets, C_indices);
}

template<size_t n>
class MeshTopologyCache{
    private:
        std::array<std::optional<size_t>, n + 1> sizes;
        std::array<std::array<std::optional<BooleanMatrixCSR>, n + 1>, n + 1> connectivity;
        std::array<std::optional<std::vector<size_t>>, n + 1> boundary;
        MeshTopologyCache(){
            this->sizes.fill(std::nullopt);
            for(auto& row : this->connectivity){
                row.fill(std::nullopt);
            }
            this->boundary.fill(std::nullopt);
        }

        template<size_t d, size_t m>
        friend class Mesh;
};

template<size_t d, size_t n>
class Mesh{
    protected:
        std::vector<Eigen::Vector<double, d>> coordinates;
        std::vector<std::array<size_t, n + 1>> cells;
        MeshTopologyCache<n> topology_cache;

        Mesh(
            std::vector<Eigen::Vector<double, d>> coordinates,
            std::vector<std::array<size_t, n + 1>> cells
        ){
            this->coordinates = coordinates;
            this->cells = cells;
            this->topology_cache = MeshTopologyCache<n>();
        }

        BooleanMatrixCSR TopConnectivityMatrix(){
            size_t number_of_cells = this->cells.size();
            size_t number_of_vertices = this->coordinates.size();
            std::vector<size_t> offsets(number_of_cells + 1);
            offsets[0] = 0;
            for(size_t i = 0; i < number_of_cells; i++){
                offsets[i + 1] = offsets[i] + n + 1;
            }
            std::vector<size_t> indices((n + 1) * number_of_cells);
            size_t k = 0;
            for(size_t i = 0; i < number_of_cells; i++){
                for(size_t j = 0; j < n + 1; j++){
                    indices[k] = this->cells[i][j];
                    k += 1;
                }
            }
            return BooleanMatrixCSR(number_of_vertices, offsets, indices);
        }

        std::pair<BooleanMatrixCSR, BooleanMatrixCSR> DecomposeTopConnectivityMatrix(size_t dimension){
            BooleanMatrixCSR connectivity = this->ConnectivityMatrix(n, 0);
            size_t number_of_cells = connectivity.NumberOfRows();
            size_t number_of_vertices = connectivity.NumberOfColumns();
            std::vector<size_t> A_offsets(number_of_cells + 1);
            std::vector<size_t> A_indices;
            std::vector<size_t> B_offsets;
            std::vector<size_t> B_indices;
            A_offsets[0] = 0;
            B_offsets.push_back(0);

            std::map<std::vector<size_t>, size_t> index_map;
            size_t k = 0;
            for(size_t i = 0; i < number_of_cells; i++){
                for(auto face : Subsets<size_t>(connectivity[i], dimension + 1)){
                    std::sort(face.begin(), face.end());
                    size_t j;
                    if(index_map.count(face)){
                        j = index_map[face];
                    }
                    else{
                        j = k;
                        index_map[face] = j;
                        B_offsets.push_back(B_offsets.back() + dimension + 1);
                        for(size_t l = 0; l < dimension + 1; l++){
                            B_indices.push_back(face[l]);
                        }
                        k += 1;
                    }
                    A_indices.push_back(j);
                }
                A_offsets[i + 1] = A_offsets[i] + BinomialCoefficient(n + 1, dimension + 1);
            }
            size_t number_of_faces = B_offsets.size() - 1;

            return {
                BooleanMatrixCSR(number_of_faces, A_offsets, A_indices),
                BooleanMatrixCSR(number_of_vertices, B_offsets, B_indices)
            };
        }

        BooleanMatrixCSR ConnectivityMatrix(size_t simplex_dimension, size_t neighbor_dimension) const{
            size_t i = simplex_dimension, j = neighbor_dimension;
            std::optional<BooleanMatrixCSR>& A = this->topology_cache.connectivity[i][j];
            if(!A.has_value()){
                if(i == n && j == 0){
                    A = this->TopConnectivityMatrix();
                }
                else if(i == j){
                    size_t size = this->Size(i);
                    A = BooleanIdentity(size);
                }
                else if(i < j){
                    A = this->ConnectivityMatrix(j, i).Transpose();
                }
                else if(i == n){
                    auto [B, C] = this->DecomposeTopConnectivityMatrix(j);
                    this->topology_cache.connectivity[j][0] = C;
                    A = B;
                }
                else if(j == 0){
                    auto [B, C] = this->DecomposeTopConnectivityMatrix(i);
                    this->topology_cache.connectivity[n][i] = B;
                    A = C;
                }
                this->topology_cache.connectivity[i][j] = A;
            }
            return *A;
        }

        std::vector<size_t> NeighborIndices(
            size_t simplex_dimension,
            size_t simplex_index,
            size_t neighbor_dimension
        ) const{
            return this->ConnectivityMatrix(simplex_dimension, neighbor_dimension)[simplex_index];
        }

        std::vector<size_t> NeighborIndices(
            size_t simplex_dimension,
            std::vector<size_t> simplex_indices,
            size_t neighbor_dimension
        ) const{
            size_t size = simplex_indices.size();
            size_t i = simplex_dimension, j = neighbor_dimension;
            std::vector<bool> is_seen(this->Size(j), false);
            std::vector<size_t> neighbor_indices;
            for(auto k : simplex_indices){
                for(auto l : NeighborIndices(i, k, j)){
                    if(!is_seen(l)){
                        neighbor_indices.push_back(l);
                        is_seen(l) = true;
                    }
                }
            }
            return neighbor_indices;
        }

        std::vector<size_t> BoundaryFacetIndices(){
            std::vector<size_t> indices;
            for(size_t i = 0; i < this->Size(n - 1); i++){
                if(NeighborIndices(n - 1, i, n).size() == 1){
                    indices.push_back(i);
                }
            }
            return indices;
        }

        std::vector<size_t> BoundaryIndices(size_t dimension) const{
            size_t k = dimension;
            std::optional<std::vector<size_t>>& indices = this->topology_cache.boundary[k];
            if(!indices.has_value()){
                if(k == n - 1){
                    indices = this->BoundaryFacetIndices();
                }
                else{
                    indices = this->NeighborIndices(n - 1, BoundaryIndices(n - 1), k);
                }
                this->topology_cache.boundary[k] = indices;
            }
            return *indices;
        }

        std::vector<size_t> VertexIndices(
            size_t simplex_dimension,
            size_t simplex_index
        ) const{
            return this->NeighborIndices(simplex_dimension, simplex_index, 0);
        }

        std::vector<Eigen::Vector<double, d>> VertexCoordinates(
            size_t simplex_dimension,
            size_t simplex_index
        ) const{
            size_t k = simplex_dimension;
            std::vector<size_t> indices = this->VertexIndices(k, simplex_index);
            std::vector<Eigen::Vector<double, d>> coordinates(k + 1);
            for(size_t i = 0; i < k + 1; i++){
                coordinates[i] = this->coordinates[indices[i]];
            }
            return coordinates;
        }

        Eigen::Matrix<double, d, n> Jacobian(size_t simplex_index) const{
            std::vector<Eigen::Vector<double, d>> coordinates = this->VertexCoordinates(n, simplex_index);
            Eigen::Matrix<double, d, n> jacobian;
            for(size_t i = 0; i < n; i++){
                jacobian.col(i) = coordinates[i + 1] - coordinates[0];
            }
            return jacobian;
        }

        friend class MeshSimplex<d, n>;

        friend class MeshSimplices<d, n>;

    public:
        size_t Size(size_t dimension){
            std::optional<size_t>& size = this->topology_cache.sizes[dimension];
            if(!size.has_value()){
                size = this->ConnectivityMatrix(dimension, 0).NumberOfRows();
                this->topology_cache.sizes[dimension] = size;
            }
            return *size;
        }
};

class UnitSquare : public Mesh<2, 2>{
    private:
        std::vector<Eigen::Vector2d> Coordinates(size_t N){
            std::vector<Eigen::Vector2d> coordinates(pow(N + 1, 2));
            size_t k = 0;
            for(size_t i = 0; i < N + 1; i++){
                for(size_t j = 0; j < N + 1; j++){
                    coordinates[k] << i / double(N), j / double(N);
                }
            }
            return coordinates;
        }

        std::vector<std::array<size_t, 3>> Cells(size_t N){
            std::vector<std::array<size_t, 3>> cells(2 * pow(N, 2));
            size_t k = 0;
            for(size_t i = 0; i < N; i++){
                for(size_t j = 0; j < N; j++){
                    cells[k] = {
                        (N + 1) * i + j,
                        (N + 1) * (i + 1) + j,
                        (N + 1) * i + j + 1
                    };
                    cells[k + 1] = {
                        (N + 1) * (i + 1) + j + 1,
                        (N + 1) * i + j + 1,
                        (N + 1) * (i + 1) + j
                    };
                    k += 2;
                }
            }
            return cells;
        }

    public:
        UnitSquare(size_t N)
            : Mesh<2, 2>(this->Coordinates(N), this->Cells(N))
        {}

        Eigen::Matrix2d Jacobian(size_t cell_index) const{
            if(cell_index % 2 == 0){
                return Eigen::Matrix2d::Identity();
            }
            else{
                return - Eigen::Matrix2d::Identity();
            }
        }
};

template<size_t d, size_t n>
class MeshSimplex{
    private:
        Mesh<d, n>& mesh;
        size_t dimension;
        size_t index;

        MeshSimplex(Mesh<d, n>& mesh, size_t dimension, size_t index){
            this->mesh = mesh;
            this->dimension = dimension;
            this->index = index;
        }

        template<size_t p, size_t m>
        friend class MeshSimplices;

    public:
        std::vector<Eigen::Vector<double, d>> VertexCoordinates() const{
            return this->mesh.VertexCoordinates(this->dimension, this->index);
        }

        Eigen::Matrix<double, d, n> Jacobian() const{
            return this->mesh.Jacobian(this->index);
        }
};

template<size_t d, size_t n>
class MeshSimplices{
    private:
        Mesh<d, n>& mesh;
        size_t dimension;
        std::vector<size_t> indices;

        MeshSimplices(Mesh<d, n>& mesh, size_t dimension, std::vector<size_t> indices){
            this->mesh = mesh;
            this->dimension = dimension;
            this->indices = indices;
        }

        friend MeshSimplices<d, n> Cells(Mesh<d, n> mesh);
        friend MeshSimplices<d, n> Boundary(Mesh<d, n> mesh, size_t dimension);
        friend MeshSimplices<d, n> Neighbors(MeshSimplex<d, n> simplex, size_t dimension);

    public:
        struct Iterator {
            const MeshSimplices* parent;
            size_t index;
            bool is_finished;

            Iterator(const MeshSimplices* parent, size_t index, bool is_finished){
                this->parent = parent;
                this->index = index;
                this->is_finished = is_finished;
            }

            const MeshSimplex<d, n> operator*() const{
                return MeshSimplex<d, n>(this->parent->mesh, this->index);
            }

            Iterator& operator++() {
                size_t i = this->index + 1;
                if(i < this->parent->mesh.Size(this->parent->dimension)){
                    this->index = i;
                }
                else{
                    this->is_finished = true;
                }
                return *this;
            }

            bool operator!=(const Iterator& other) const {
                return this->is_finished != other.is_finished;
            }
        };

        Iterator begin() const{
            return Iterator(this, 0, false);
        }

        Iterator end() const{
            return Iterator(this, 0, true);
        }
};

template<size_t d, size_t n>
MeshSimplices<d, n> Cells(Mesh<d, n> mesh){
    size_t number_of_cells = mesh.Size(n);
    std::vector<size_t> indices(number_of_cells);
    for(size_t i = 0; i < number_of_cells; i++){
        indices[i] = i;
    }
    return MeshSimplices<d, n>(mesh, n, indices);
}

template<size_t d, size_t n>
MeshSimplices<d, n> Neighbors(MeshSimplex<d, n> simplex, size_t dimension){
    std::vector<size_t> indices = simplex.mesh.NeighborIndices(simplex.dimension, simplex.index, dimension);
    return MeshSimplices<d, n>(simplex.mesh, simplex.dimension, indices);
}

int main(){
    Mesh<2, 2> mesh = UnitSquare(1);
}