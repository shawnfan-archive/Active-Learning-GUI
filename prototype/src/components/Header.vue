<template>
  <div>
    <header>{{ greet() }}</header>
  </div>
</template>

<script>
export default {
  data() {
    return {
      username: 'User1'
    }
  },
  methods: {
    greet: function() {
      return 'Welcome to the Active Learning GUI, ' + this.username + '!'
    } 
  }
}
</script>