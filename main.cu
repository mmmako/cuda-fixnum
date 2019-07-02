#include <cstdio>
#include <cstring>
#include <cassert>
#include <vector>

#include "fixnum/warp_fixnum.cu"
#include "array/fixnum_array.h"
#include "functions/modexp.cu"
#include "functions/multi_modexp.cu"
#include "modnum/modnum_monty_redc.cu"
#include "modnum/modnum_monty_cios.cu"

const unsigned int bytes_per_elem = 128;
const unsigned int io_bytes_per_elem = 96;


using namespace std;
using namespace cuFIXNUM;

template< typename fixnum, typename modnum>
__device__ void mul_quad(fixnum &r0, fixnum &r1, fixnum a0, fixnum a1, fixnum b0, fixnum b1, modnum mod, fixnum non_residue) {
    //fixnum a0b0, a1b1, a1b1nr;

    //mod.mul(a0b0, a0, b0);
    //mod.mul(a1b1, a1, b1);
    //mod.mul(a1b1nr, a1b1, non_residue);

    //fixnum a0pb0, a1pb1;
    //mod.add(a0pb0, a0, b0);
    //mod.add(a1pb1, a1, b1);

    //fixnum prod, sub1;
    //mod.mul(prod, a0pb0, a1pb1);
    //mod.sub(sub1, prod, a0b0);

    fixnum a0b0, a0b1, a1b0, a1b1, a1b1nr;
    mod.mul(a0b0, a0, b0);
    mod.mul(a0b1, a0, b1);
    mod.mul(a1b0, a1, b0);
    mod.mul(a1b1, a1, b1);
    mod.mul(a1b1nr, a1b1, non_residue);

    fixnum s0, s1;
    //mod.add(s0, a0b0, a1b1nr);
    //mod.sub(s1, sub1, a1b1);
    mod.add(s0, a0b0, a1b1nr);
    mod.add(s1, a0b1, a1b0);

    r0 = s0;
    r1 = s1;
}

template< typename fixnum >
struct mul_quad_and_convert {
  // redc may be worth trying over cios
  typedef modnum_monty_cios<fixnum> modnum;
  __device__ void operator()(fixnum &r0, fixnum &r1, fixnum a0, fixnum a1, fixnum b0, fixnum b1, fixnum my_mod, fixnum non_residue) {
      modnum mod = modnum(my_mod);

      fixnum sm0, sm1;

      fixnum am0, am1;
      fixnum bm0, bm1;
      fixnum non_residuem;
      mod.to_modnum(am0, a0);
      mod.to_modnum(am1, a1);
      mod.to_modnum(bm0, b0);
      mod.to_modnum(bm1, b1);
      mod.to_modnum(non_residuem, non_residue);
      
      mul_quad(sm0, sm1, am0, am1, bm0, bm1, mod, non_residuem);

      fixnum s0, s1;
      mod.from_modnum(s0, sm0);
      mod.from_modnum(s1, sm1);

      r0 = s0;
      r1 = s1;
  }
};

template< int fn_bytes, typename fixnum_array >
void print_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    uint8_t local_results[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);

    for (int i = 0; i < lrl; i++) {
      printf("%i ", local_results[i]);
    }
    printf("\n");
}

template< int fn_bytes, typename fixnum_array >
vector<uint8_t*> get_fixnum_array(fixnum_array* res, int nelts) {
    int lrl = fn_bytes*nelts;
    //uint8_t local_results[lrl];
    uint8_t* local_results = new uint8_t[lrl];
    int ret_nelts;
    for (int i = 0; i < lrl; i++) {
      local_results[i] = 0;
    }
    res->retrieve_all(local_results, fn_bytes*nelts, &ret_nelts);
    vector<uint8_t*> res_v;
    for (int n = 0; n < nelts; n++) {
      uint8_t* a = (uint8_t*)malloc(fn_bytes*sizeof(uint8_t));
      for (int i = 0; i < fn_bytes; i++) {
        a[i] = local_results[n*fn_bytes + i];
      }
      res_v.emplace_back(a);
    }
    delete[](local_results);
    return res_v;
}


template< int fn_bytes, typename word_fixnum, template <typename> class Func >
std::pair<std::vector<uint8_t*>, std::vector<uint8_t*>> compute_product(std::vector<uint8_t*> a0, std::vector<uint8_t*> a1,
        std::vector<uint8_t*> b0, std::vector<uint8_t*> b1, uint8_t* input_m_base, uint8_t* non_residue) {
    typedef warp_fixnum<fn_bytes, word_fixnum> fixnum;
    typedef fixnum_array<fixnum> fixnum_array;

    int nelts = a0.size();

    uint8_t *input_a0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_a1 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_a0[i] = a0[i/fn_bytes][i%fn_bytes];
      input_a1[i] = a1[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_b0 = new uint8_t[fn_bytes * nelts];
    uint8_t *input_b1 = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_b0[i] = b0[i/fn_bytes][i%fn_bytes];
      input_b1[i] = b1[i/fn_bytes][i%fn_bytes];
    }

    uint8_t *input_m = new uint8_t[fn_bytes * nelts];
    uint8_t *input_nr = new uint8_t[fn_bytes * nelts];
    for (int i = 0; i < fn_bytes * nelts; ++i) {
      input_m[i] = input_m_base[i%fn_bytes];
      input_nr[i] = non_residue[i%fn_bytes];
    }

    // TODO reuse modulus as a constant instead of passing in nelts times
    fixnum_array *res0, *res1, *in_a0, *in_a1, *in_b0, *in_b1, *inM, *inNR;
    in_a0 = fixnum_array::create(input_a0, fn_bytes * nelts, fn_bytes);
    in_a1 = fixnum_array::create(input_a1, fn_bytes * nelts, fn_bytes);
    in_b0 = fixnum_array::create(input_b0, fn_bytes * nelts, fn_bytes);
    in_b1 = fixnum_array::create(input_b1, fn_bytes * nelts, fn_bytes);
    inM = fixnum_array::create(input_m, fn_bytes * nelts, fn_bytes);
    inNR = fixnum_array::create(input_nr, fn_bytes * nelts, fn_bytes);
    res0 = fixnum_array::create(nelts);
    res1 = fixnum_array::create(nelts);

    fixnum_array::template map<Func>(res0, res1, in_a0, in_a1, in_b0, in_b1, inM, inNR);

    vector<uint8_t*> v_res0 = get_fixnum_array<fn_bytes, fixnum_array>(res0, nelts);
    vector<uint8_t*> v_res1 = get_fixnum_array<fn_bytes, fixnum_array>(res1, nelts);

    //TODO to do stage 1 field arithmetic, instead of a map, do a reduce

    delete in_a0;
    delete in_a1;
    delete in_b0;
    delete in_b1;
    delete inM;
    delete res0;
    delete res1;
    delete[] input_a0;
    delete[] input_a1;
    delete[] input_b0;
    delete[] input_b1;
    delete[] input_m;
    return std::make_pair(v_res0, v_res1);
}

uint8_t* read_mnt_fq(FILE* inputs) {
  uint8_t* buf = (uint8_t*)calloc(bytes_per_elem, sizeof(uint8_t));
  // the input is montgomery representation x * 2^768 whereas cuda-fixnum expects x * 2^1024 so we shift over by (1024-768)/8 bytes
  fread((void*)(buf), io_bytes_per_elem*sizeof(uint8_t), 1, inputs);
  return buf;
}

void write_mnt_fq(uint8_t* fq, FILE* outputs) {
  fwrite((void *) fq, io_bytes_per_elem * sizeof(uint8_t), 1, outputs);
}

void print_array(uint8_t* a) {
  for (int j = 0; j < 128; j++) {
    printf("%x ", ((uint8_t*)(a))[j]);
  }
  printf("\n");
}

int main(int argc, char* argv[]) {
  setbuf(stdout, NULL);

  // mnt4_q
  uint8_t mnt4_modulus[bytes_per_elem] = {1,128,94,36,222,99,144,94,159,17,221,44,82,84,157,227,240,37,196,154,113,16,136,99,164,84,114,118,233,204,90,104,56,126,83,203,165,13,15,184,157,5,24,242,118,231,23,177,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  uint8_t non_residue[bytes_per_elem] = {13,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  // mnt6_q
  //uint8_t mnt6_modulus[bytes_per_elem] = {1,0,0,64,226,118,7,217,79,58,161,15,23,153,160,78,151,87,0,63,188,129,195,214,164,58,153,52,118,249,223,185,54,38,33,41,148,202,235,62,155,169,89,200,40,92,108,178,157,247,90,161,217,36,209,153,141,237,160,232,37,185,253,7,115,216,151,108,249,232,183,94,237,175,143,91,80,151,249,183,173,205,226,238,34,144,34,16,17,196,146,45,198,196,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};

  auto inputs = fopen(argv[2], "r");
  auto outputs = fopen(argv[3], "w");

  size_t n;

   while (true) {
    size_t elts_read = fread((void *) &n, sizeof(size_t), 1, inputs);
    if (elts_read == 0) { break; }

    std::vector<uint8_t*> x0;
    std::vector<uint8_t*> x1;
    for (size_t i = 0; i < n; ++i) {
      x0.emplace_back(read_mnt_fq(inputs));
      x1.emplace_back(read_mnt_fq(inputs));
    }

    std::vector<uint8_t*> y0;
    std::vector<uint8_t*> y1;
    for (size_t i = 0; i < n; ++i) {
      y0.emplace_back(read_mnt_fq(inputs));
      y1.emplace_back(read_mnt_fq(inputs));
    }

    std::pair<std::vector<uint8_t*>, std::vector<uint8_t*>> res_x
        = compute_product<bytes_per_elem, u64_fixnum, mul_quad_and_convert>(x0, x1, y0, y1, mnt4_modulus, non_residue);

    for (size_t i = 0; i < n; ++i) {
      write_mnt_fq(res_x.first[i], outputs);
      write_mnt_fq(res_x.second[i], outputs);
    }

    for (size_t i = 0; i < n; ++i) {
      free(x0[i]);
      free(x1[i]);
      free(y0[i]);
      free(y1[i]);
      free(res_x.first[i]);
      free(res_x.second[i]);
    }
  }

  return 0;
}

