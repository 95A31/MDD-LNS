/* -*- mode: C++; c-basic-offset: 2; indent-tabs-mode: nil -*- */
/*
 *  Main author:
 *     Christian Schulte <schulte@gecode.org>
 *
 *  Copyright:
 *     Christian Schulte, 2012
 *
 *  This file is part of Gecode, the generic constraint
 *  development environment:
 *     http://www.gecode.org
 *
 *  Permission is hereby granted, free of charge, to any person obtaining
 *  a copy of this software and associated documentation files (the
 *  "Software"), to deal in the Software without restriction, including
 *  without limitation the rights to use, copy, modify, merge, publish,
 *  distribute, sublicense, and/or sell copies of the Software, and to
 *  permit persons to whom the Software is furnished to do so, subject to
 *  the following conditions:
 *
 *  The above copyright notice and this permission notice shall be
 *  included in all copies or substantial portions of the Software.
 *
 *  THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 *  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 *  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 *  NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 *  LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 *  OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 *  WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
 *
 */

namespace Gecode {

  /**
   * \defgroup TaskBranchValCommit Generic value commit for brancher based on view and value selection
   *
   * \ingroup TaskBranchViewVal
   */
  //@{
  /// Base class for value commit
  template<class View_, class Val_>
  class ValCommit {
  public:
    /// View type
    typedef View_ View;
    /// Corresponding variable type
    typedef typename View::VarType Var;
    /// Value type
    typedef Val_ Val;
  public:
    /// Constructor for initialization
    ValCommit(Space& home, const ValBranch<Var>& vb);
    /// Constructor for cloning
    ValCommit(Space& home, ValCommit<View,Val>& vs);
    /// Whether dispose must always be called (that is, notice is needed)
    bool notice(void) const;
    /// Delete value commit
    void dispose(Space& home);
  };

  /// Class for user-defined value commit
  template<class View>
  class ValCommitFunction : public
  ValCommit<View,
            typename BranchTraits<typename View::VarType>::ValType> {
    typedef typename ValCommit<View,
                               typename BranchTraits<typename View::VarType>
                                 ::ValType>::Val Val;
  public:
    /// The corresponding variable type
    typedef typename View::VarType Var;
    /// The corresponding commit function
    typedef typename BranchTraits<Var>::Commit CommitFunction;
  protected:
    /// The user-defined commit function
    SharedData<CommitFunction> c;
  public:
    /// Constructor for initialization
    ValCommitFunction(Space& home, const ValBranch<Var>& vb);
    /// Constructor for cloning during copying
    ValCommitFunction(Space& home, ValCommitFunction& vc);
    /// Perform user-defined commit
    ModEvent commit(Space& home, unsigned int a, View x, int i, Val n);
    /// Create no-good literal for alternative \a a
    NGL* ngl(Space& home, unsigned int a, View x, Val n) const;
    /// Print on \a o the alternative \a with view \a x at position \a i and value \a n
    void print(const Space& home, unsigned int a, View x, int i,
               const Val& n, std::ostream& o) const;
    /// Whether dispose must always be called (that is, notice is needed)
    bool notice(void) const;
    /// Delete value commit
    void dispose(Space& home);
  };
  //@}

  // Baseclass for value commit
  template<class View, class Val>
  forceinline
  ValCommit<View,Val>::ValCommit(Space&, const ValBranch<Var>&) {}
  template<class View, class Val>
  forceinline
  ValCommit<View,Val>::ValCommit(Space&, ValCommit<View,Val>&) {}
  template<class View, class Val>
  forceinline bool
  ValCommit<View,Val>::notice(void) const {
    return false;
  }
  template<class View, class Val>
  forceinline void
  ValCommit<View,Val>::dispose(Space&) {}


  // User-defined value selection
  template<class View>
  forceinline
  ValCommitFunction<View>::ValCommitFunction(Space& home,
                                             const ValBranch<Var>& vb)
    : ValCommit<View,Val>(home,vb), c(vb.commit()) {
    if (!c())
      throw InvalidFunction("ValCommitFunction::ValCommitFunction");
  }
  template<class View>
  forceinline
  ValCommitFunction<View>::ValCommitFunction(Space& home,
                                             ValCommitFunction<View>& vc)
    : ValCommit<View,Val>(home,vc), c(vc.c) {
  }
  template<class View>
  forceinline ModEvent
  ValCommitFunction<View>::commit(Space& home, unsigned int a, View x, int i,
                                  Val n) {
    typename View::VarType y(x.varimp());
    GECODE_VALID_FUNCTION(c());
    c()(home,a,y,i,n);
    return home.failed() ? ES_FAILED : ES_OK;
  }
  template<class View>
  forceinline NGL*
  ValCommitFunction<View>::ngl(Space&, unsigned int, View, Val) const {
    return nullptr;
  }
  template<class View>
  forceinline void
  ValCommitFunction<View>::print(const Space&, unsigned int,
                                 View, int i, const Val&,
                                 std::ostream& o) const {
    o << "var[" << i << "] is user-defined.";
  }
  template<class View>
  forceinline bool
  ValCommitFunction<View>::notice(void) const {
    return true;
  }
  template<class View>
  forceinline void
  ValCommitFunction<View>::dispose(Space&) {
    c.~SharedData<CommitFunction>();
  }

}

// STATISTICS: kernel-branch
