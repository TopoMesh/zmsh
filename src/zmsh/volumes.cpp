#include <Eigen/Core>
#include <Eigen/Dense>
#include <boost/numeric/interval.hpp>
#include <boost/numeric/interval/utility.hpp>
#include <boost/multiprecision/eigen.hpp>
#include <gmpxx.h>
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <stdexcept>

/**
 * Without this truly unsavory hack, we get a compile error to the effect of
 * `ambiguous overload for 'operator*`. I apologize.
 */
namespace std {
    template <typename T, typename P>
    using interval = boost::numeric::interval<T, P>;

    template <typename X, typename S, typename P>
    struct is_convertible<X, interval<S, P>> {
        enum { value = is_convertible<X, S>::value };
    };

    template <typename S, typename P1, typename P2>
    struct is_convertible<interval<S, P1>, interval<S, P2>> {
        enum { value = true };
    };
} // namespace std


/**
 * Shims to make Eigen work with GMP rationals
 */
namespace Eigen {
    template<> struct NumTraits<mpq_class> : GenericNumTraits<mpq_class> {
        typedef mpq_class Real;
        typedef mpq_class NonInteger;
        typedef mpq_class Nested;

        static inline Real epsilon() { return 0; }
        static inline Real dummy_precision() { return 0; }
        static inline int digits10() { return 0; }

        enum {
            IsInteger = 0,
            IsSigned = 1,
            IsComplex = 0,
            RequireInitialization = 1,
            ReadCost = 6,
            AddCost = 150,
            MulCost = 100
        };
    };
}


namespace predicates {
    namespace internal {
        template <typename T>
        using Matrix = Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic>;

        template <typename T>
        T volume(const Eigen::MatrixXd& zs) {
            const size_t m = zs.rows();
            const size_t n = zs.cols();

            if (n != m + 1)
                throw std::invalid_argument("Wrong matrix size!");

            Matrix<T> ws(n, n);
            ws.row(0).setConstant(1);
            ws.block(1, 0, m, n) = zs.cast<T>();
            return ws.determinant();
        }
    }

    /**
     * Sign-exact determinant computation on Eigen matrices using intervals as
     * a first resort and rationals as a fallback.
     */
    double volume(const Eigen::MatrixXd& zs) {
        using Rational = mpq_class;
        using Interval = boost::numeric::interval<double>;

        try {
            const auto result = internal::volume<Interval>(zs);
            if (not boost::numeric::zero_in(result))
                return boost::numeric::median(result);
        }
        catch (const boost::numeric::interval_lib::comparison_error& e) {}

        return internal::volume<Rational>(zs).get_d();
    }
}


PYBIND11_MODULE(volumes, m) {
    m.doc() = R"pbdoc(Computing signed volumes)pbdoc";
    m.def(
        "volume",
        [](pybind11::EigenDRef<Eigen::MatrixXd> m){
            return predicates::volume(m);
        }
    );
}
