extern crate csv;
extern crate ndarray;
extern crate ndarray_csv;
use std::ops::AddAssign;

use counter::Counter;
use csv::WriterBuilder;
use ndarray::{s, Array, Array1, Array2, ArrayView1, ArrayView2};
use ndarray_csv::Array2Writer;
use ndarray_rand::rand_distr::Uniform;
use ndarray_rand::RandomExt;
use std::error::Error;
use std::fs::File;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub type Matrix = Array2<f64>;
pub type MatrixView<'a> = ArrayView2<'a, f64>;
pub type Vector = Array1<usize>;

pub struct KMeansState {
    centroids: Matrix,
    cluster_assignment: Vector,
}

pub fn generate_random_matrix(nrows: usize, ncols: usize) -> Matrix {
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

fn assign_to_clusters(data: &MatrixView, state: &mut KMeansState) {
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
}

pub fn print_state(state: &KMeansState, data: &MatrixView) {
    println!("Centroids are: {}", state.centroids);
    for (i, row) in data.rows().into_iter().enumerate() {
        println!("{} -> cluster {}", row, state.cluster_assignment[i]);
    }
}

// pub fn to_csv<'a>(state: &KMeansState, data: &MatrixView<'a>) -> Result<(), Box<dyn Error>> {
//     let filenames = vec!["data.txt", "centroids.txt", "clusters.txt"];
//     let assignments2d = state
//         .cluster_assignment
//         .mapv(|elem| elem as f64) // np.astype(float64)
//         .into_shape((1, data.nrows()))
//         .unwrap();
//     let matrices: Vec<&MatrixView> = vec![&data, &state.centroids.view(), &assignments2d.view()];

//     for (filename, values) in filenames.iter().zip(matrices) {
//         let file = File::create(filename)?;
//         let mut writer = WriterBuilder::new().has_headers(false).from_writer(file);
//         writer.serialize_array2(&values.to_owned())?;
//     }
//     Ok(())
// }

pub fn update(state: &mut KMeansState, data: &MatrixView) {
    // find the nearest centroid for each data point (row)
    for i in 1..5 {
        assign_to_clusters(data, state);

        // update centroids -> TODO this can overflow really fast, see running avg implementation in
        // https://github.com/rust-ndarray/ndarray-examples/blob/master/k_means/src/lib.rs#L173
        let mut new_centroids: Matrix = Array2::zeros(state.centroids.dim());
        let cluster_member_counts = state.cluster_assignment.iter().collect::<Counter<_>>();
        for (i, row) in data.rows().into_iter().enumerate() {
            new_centroids
                .row_mut(state.cluster_assignment[i])
                .add_assign(&row); // not sure why += doesn't work here but /= works below
        }
        for (i, mut centroid) in new_centroids.rows_mut().into_iter().enumerate() {
            centroid /= cluster_member_counts[&i] as f64;
        }
        state.centroids = new_centroids;
    }
}

fn init_state<'a>(data: &'a MatrixView<'a>, k: &'a usize) -> KMeansState {
    let mut centroids = Array2::<f64>::zeros((k.to_owned(), data.ncols()));
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

pub fn cluster<'a>(data: &'a MatrixView<'a>, k: &'a usize) -> Vector {
    let mut state = init_state(&data, k);
    update(&mut state, &data);
    state.cluster_assignment
}

#[cfg(test)]
mod tests {
    use super::*;
    const ROWS: usize = 10; // rows
    const COLS: usize = 2; // cols
    const K: usize = 3; // num clusters

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);

        let data = generate_random_matrix(ROWS, COLS);
        let mut state = init_state(&data.view(), &K);
        update(&mut state, &data.view());
        // to_csv(&state, &data);
    }
}
