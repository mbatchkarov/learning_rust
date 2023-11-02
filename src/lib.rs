use std::ops::{AddAssign, DivAssign};

use counter::Counter;
use ndarray::{s, Array, Array1, Array2, ArrayView1};
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

const K: usize = 3; // num clusters

pub struct KMeansState {
    centroids: Array2<f64>,
    cluster_assignment: Array1<usize>,
}

pub fn generate_random_matrix(nrows: usize, ncols: usize) -> Array2<f64> {
    let mut data = Array::random((nrows, ncols), Uniform::new(0., 5.));
    // make the clusters more visible when plotted
    for i in 0..(nrows / 3) {
        data[(i, 0)] += 10.0;
    }
    for i in (2 * nrows / 3)..nrows {
        data[(i, 1)] += 10.0;
    }
    return data;
}

fn euclidean_dist(v1: &ArrayView1<f64>, v2: &ArrayView1<f64>) -> f64 {
    // compute sum((a - b) ^ 2)- don't need sqrt because we only care about argmin
    return (v1 - v2).map(|a| a.powi(2)).sum(); // TODO in parallel? SIMD?
}

fn assing_to_clusters(data: &Array2<f64>, state: &mut KMeansState) {
    // TODO compute pairwise distances in parallel, save to matrix, then get argmax by row
    for (i, row) in data.rows().into_iter().enumerate() {
        let mut dist = std::f64::MAX;

        for (j, centroid) in state.centroids.rows().into_iter().enumerate() {
            let dist_to_this = euclidean_dist(&row, &centroid);
            if dist_to_this < dist {
                dist = dist_to_this;
                state.cluster_assignment[i] = j;
            }
        }
    }
    println!("Cluster assignment {}", state.cluster_assignment);
}

pub fn print_state(state: &KMeansState, data: &Array2<f64>) {
    println!("Centroids are: {}", state.centroids);
    for (i, row) in data.rows().into_iter().enumerate() {
        println!("{} -> cluster {}", row, state.cluster_assignment[i]);
    }
}

pub fn update(state: &mut KMeansState, data: &Array2<f64>) {
    // find the nearest centroid for each data point (row)
    // let distances: Array1<f64> = Array1::zeros(R) * -1.0;
    println!("State at start");
    print_state(&state, data);
    for i in 1..5 {
        assing_to_clusters(data, state);
        println!("State at iteration {}", i);
        print_state(&state, data)
    }

    // update centroids -> TODO this can overflow really fast, see running avg implementation in
    // https://github.com/rust-ndarray/ndarray-examples/blob/master/k_means/src/lib.rs#L173
    let mut new_centroids: Array2<f64> = Array2::zeros(state.centroids.dim());
    let cluster_member_counts = state.cluster_assignment.iter().collect::<Counter<_>>();
    for (i, row) in data.rows().into_iter().enumerate() {
        new_centroids
            .row_mut(state.cluster_assignment[i])
            .add_assign(&row);
    }
    for (i, mut centroid) in new_centroids.rows_mut().into_iter().enumerate() {
        centroid.div_assign(cluster_member_counts[&i] as f64);
    }
    state.centroids = new_centroids;
}

pub fn init_state(data: &Array2<f64>) -> KMeansState {
    let mut centroids = Array2::<f64>::zeros((K, data.ncols()));
    // take the first 3 elements "randomly"
    // data.slice(s![..K, ..])
    //     .assign_to(centroids.slice_mut(s![..K, ..]));

    // take one centroid from each cluster so iteration is guaranteed to converge
    for (i, j) in vec![0, data.nrows() / 2, data.nrows() - 1]
        .into_iter()
        .enumerate()
    {
        data.slice(s![j, ..])
            .assign_to(centroids.slice_mut(s![i, ..]));
    }

    return KMeansState {
        // centroids: data.slice(Slice::from(0..=K)), // TODO need 0..K
        centroids: centroids,
        cluster_assignment: Array1::zeros(data.nrows()),
    };
}

#[cfg(test)]
mod tests {
    use super::*;
    const ROWS: usize = 10; // rows
    const COLS: usize = 2; // cols

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);

        let data = generate_random_matrix(ROWS, COLS);
        let mut state = init_state(&data);
        update(&mut state, &data);
    }
}
